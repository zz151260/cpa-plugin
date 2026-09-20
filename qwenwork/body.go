// body.go constructs the QoderWork agent_chat_generation request body from
// OpenAI-style chat completion inputs.
//
// The base template lives in baseprompt.json (embedded). Per-request we
// overwrite request/session ids, timestamps, model key, and the user prompt.
package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

//go:embed baseprompt.json
var basepromptJSON []byte

// cpaToUpstreamKey maps CPA-facing model names to the tier keys QwenWorkCN
// recognises. The gateway routes tiers, not concrete models — the authoritative
// list comes from GET /algo/api/v2/model/list, whose "qwork" array contains
// exactly three entries (verified 2026-09-18, matching the desktop client's
// picker):
//
//	pro                  高级                1X    default, balanced
//	flash                标准｜Qwen3.8-Flash  0.1X  fastest
//	qwen3.8-max-preview  Qwen3.8-Max         1.8X  strongest
//
// The first two accept friendly aliases so callers can use readable names.
// Note the older "qwork-advanced"/"qwork-lite" names were plugin inventions:
// upstream never used them, and "qwork-lite" was never routable at all.
func cpaToUpstreamKey(cpaModel string) string {
	switch cpaModel {
	case "qwork-pro", "advanced", "qwork-advanced", "qwenwork-auto":
		return "pro"
	case "qwork-flash", "standard", "qwen3.8-flash":
		return "flash"
	case "qwork-max", "qwen3.8-max":
		return "qwen3.8-max-preview"
	}
	return cpaModel
}

// imagePart is one image carried by a user message. Either Base64 (with
// MediaType) or URL is set.
type imagePart struct {
	Base64    string
	MediaType string
	URL       string
}

// openAIMessage is one message in the chat completion format. Content accepts
// both shapes seen in practice:
//
//	"content": "text"                                  (OpenAI chat completions)
//	"content": [{"type":"text","text":"…"}]            (Anthropic messages)
//
// The Anthropic form is what the CPA /v1/messages front end forwards, so a
// plain string field here makes every such request fail with
// "cannot unmarshal array into Go struct field".
//
// Text parts are concatenated into Content; image parts are kept in Images so
// buildQwenBody can forward them upstream (the gateway takes images separately
// from the prompt text).
type openAIMessage struct {
	Role    string
	Content string
	Images  []imagePart
}

// UnmarshalJSON decodes either a bare string or an array of typed content parts.
// Text parts are concatenated; image parts are collected. Unknown part types are
// skipped rather than erroring, so an unsupported part does not sink an
// otherwise valid prompt.
func (m *openAIMessage) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	if len(raw.Content) == 0 {
		return nil
	}
	// String form.
	var s string
	if err := json.Unmarshal(raw.Content, &s); err == nil {
		m.Content = s
		return nil
	}
	// Array-of-parts form.
	var parts []struct {
		Type   string `json:"type"`
		Text   string `json:"text"`
		Source struct {
			Type      string `json:"type"` // "base64" | "url"
			MediaType string `json:"media_type"`
			Data      string `json:"data"`
			URL       string `json:"url"`
		} `json:"source"`
		// OpenAI-style image part
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return fmt.Errorf("message content must be a string or an array of parts: %w", err)
	}
	var sb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "image", "image_url":
			img := imagePart{MediaType: p.Source.MediaType, URL: p.Source.URL}
			switch p.Source.Type {
			case "base64":
				img.Base64 = p.Source.Data
			case "url":
				img.URL = p.Source.URL
			}
			// OpenAI-shaped part carries the URL here instead.
			if img.URL == "" && p.ImageURL.URL != "" {
				img.URL = p.ImageURL.URL
			}
			if img.Base64 != "" || img.URL != "" {
				m.Images = append(m.Images, img)
			}
		default:
			if p.Text != "" {
				sb.WriteString(p.Text)
			}
		}
	}
	m.Content = sb.String()
	return nil
}

// openAIRequest is the CPA-facing chat completion request.
type openAIRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

// extractLatestUserPrompt returns the content of the last user message.
// extractLatestUserPrompt returns the text of the newest user message that has
// any. It walks backwards rather than returning the last user message blindly,
// because an image-only turn would otherwise yield "" and leave the gateway's
// prompt fields empty.
func extractLatestUserPrompt(messages []openAIMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && messages[i].Content != "" {
			return messages[i].Content
		}
	}
	return ""
}

// requestHasImages reports whether any message carries an image part.
func requestHasImages(messages []openAIMessage) bool {
	for _, m := range messages {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// buildQwenBody renders the upstream agent_chat_generation body for one request.
// modelKey is the upstream key (already mapped via cpaToUpstreamKey).
func buildQwenBody(req *openAIRequest, modelKey, userType string) ([]byte, error) {
	var base map[string]any
	if err := json.Unmarshal(basepromptJSON, &base); err != nil {
		return nil, fmt.Errorf("baseprompt decode: %w", err)
	}

	prompt := extractLatestUserPrompt(req.Messages)
	// An image-only turn has no text, which is valid multimodal input — only
	// reject when there is neither text nor an image anywhere in the request.
	if prompt == "" && !requestHasImages(req.Messages) {
		return nil, fmt.Errorf("no user message in request")
	}

	nid := uuid.NewString()
	base["request_id"] = nid
	base["chat_record_id"] = nid
	base["request_set_id"] = uuid.NewString()
	base["session_id"] = uuid.NewString()
	base["stream"] = true
	base["aliyun_user_type"] = userType
	base["agent_id"] = "agent_common"

	// model_config
	if mc, ok := base["model_config"].(map[string]any); ok {
		mc["key"] = modelKey
	}

	// chat_context.text.text + chat_context.extra.originalContent.text
	if cc, ok := base["chat_context"].(map[string]any); ok {
		if txt, ok := cc["text"].(map[string]any); ok {
			txt["text"] = prompt
		}
		if extra, ok := cc["extra"].(map[string]any); ok {
			if oc, ok := extra["originalContent"].(map[string]any); ok {
				oc["text"] = prompt
			}
			if mc, ok := extra["modelConfig"].(map[string]any); ok {
				mc["key"] = modelKey
			}
		}
	}

	// messages: keep system prompt from baseprompt (template has it), replace user/assistant
	var systemMsgs []any
	if msgs, ok := base["messages"].([]any); ok {
		for _, m := range msgs {
			if mm, ok := m.(map[string]any); ok {
				if role, _ := mm["role"].(string); role == "system" {
					systemMsgs = append(systemMsgs, m)
				}
			}
		}
	}
	// Append the actual conversation. Messages carrying images also get a
	// "contents" array, which is how the gateway receives multimodal input;
	// text-only messages keep the plain "content" shape the template uses.
	var allImages []map[string]any
	for _, m := range req.Messages {
		entry := map[string]any{
			"role":    m.Role,
			"content": m.Content,
		}
		if len(m.Images) > 0 {
			contents := make([]map[string]any, 0, len(m.Images)+1)
			if m.Content != "" {
				contents = append(contents, map[string]any{"type": "text", "text": m.Content})
			}
			for _, img := range m.Images {
				src := map[string]any{"type": "base64", "media_type": img.MediaType, "data": img.Base64}
				if img.Base64 == "" {
					src = map[string]any{"type": "url", "url": img.URL}
				}
				contents = append(contents, map[string]any{"type": "image", "source": src})
				allImages = append(allImages, map[string]any{"source": src})
			}
			entry["contents"] = contents
		}
		systemMsgs = append(systemMsgs, entry)
	}
	base["messages"] = systemMsgs

	// Top-level image_urls mirrors the images of this turn; the gateway reads it
	// in addition to the per-message contents.
	if len(allImages) > 0 {
		base["image_urls"] = allImages
		if cc, ok := base["chat_context"].(map[string]any); ok {
			cc["imageUrls"] = allImages
		}
	}

	// business
	if biz, ok := base["business"].(map[string]any); ok {
		biz["id"] = uuid.NewString()
		biz["begin_at"] = time.Now().UnixMilli()
		if len(prompt) > 30 {
			biz["name"] = prompt[:30]
		} else {
			biz["name"] = prompt
		}
	}

	return json.Marshal(base)
}
