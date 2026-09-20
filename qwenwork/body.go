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
	// Tools are forwarded verbatim. The agent harness defines them and decides
	// which the model may call; substituting a fixed list makes every tool the
	// model actually wants (Agent, AskUserQuestion, …) unavailable, so the model
	// answers with prose instead of a tool call and the agent loop ends early.
	Tools      []json.RawMessage `json:"tools"`
	ToolChoice json.RawMessage   `json:"tool_choice"`
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
	base["request_set_id"] = nid
	base["session_id"] = uuid.NewString()
	base["stream"] = true
	base["aliyun_user_type"] = userType
	base["agent_id"] = "agent_common"
	base["task_id"] = "common"
	// The gateway expects the literal string "3" here (not a number); the
	// upstream reference sends version:"3" alongside session_type qoder_work.
	base["version"] = "3"
	base["session_type"] = "qoder_work"

	// model_config: mirror the reference implementation's flags, notably is_vl
	// (the gateway advertises image input only when this is true).
	if mc, ok := base["model_config"].(map[string]any); ok {
		mc["key"] = modelKey
		mc["display_name"] = modelKey
		mc["is_vl"] = true
		mc["max_input_tokens"] = 180000
	}

	// chat_context: the upstream reference uses a flat text string plus an
	// is_vl:true modelConfig and originalContent string; keep that shape so the
	// gateway's multimodal path engages.
	if cc, ok := base["chat_context"].(map[string]any); ok {
		cc["text"] = prompt
		cc["chatPrompt"] = ""
		if extra, ok := cc["extra"].(map[string]any); ok {
			extra["originalContent"] = prompt
			if mc, ok := extra["modelConfig"].(map[string]any); ok {
				mc["key"] = modelKey
				mc["is_vl"] = true
				mc["is_reasoning"] = false
			}
		}
	}

	// messages: start from the template's system message, then merge in any
	// system message the caller sent (agent harnesses put their operating
	// instructions there), and finally append a narration nudge.
	//
	// Why the nudge: Qwen3.8-Flash tends to reply with reasoning + tool calls
	// and no visible text, which reads as "worked a bit then stopped" in agent
	// UIs that show only assistant prose between turns. Asking for a one-line
	// intent statement before acting keeps the loop legible for the user.
	var systemParts []string
	if msgs, ok := base["messages"].([]any); ok {
		kept := make([]any, 0, len(msgs))
		for _, m := range msgs {
			mm, ok := m.(map[string]any)
			if !ok {
				kept = append(kept, m)
				continue
			}
			if role, _ := mm["role"].(string); role == "system" {
				if s, _ := mm["content"].(string); strings.TrimSpace(s) != "" {
					systemParts = append(systemParts, s)
				}
				continue // replaced below by the merged system message
			}
			kept = append(kept, m)
		}
		base["messages"] = kept
	}
	for _, m := range req.Messages {
		if m.Role == "system" && strings.TrimSpace(m.Content) != "" {
			systemParts = append(systemParts, m.Content)
		}
	}
	systemParts = append(systemParts,
		"Before each tool call, state your intent in one short sentence in the user's language (e.g. \"I'll list the directory first\"). "+
			"After finishing a multi-step task, summarise what was done in 2-3 sentences. Never stay silent between actions. "+
			"IMPORTANT — keep working until the task is actually finished: when you still have steps left, or when you just described what to do next, "+
			"emit the tool call for that next step in the SAME reply instead of ending your turn with commentary. "+
			"End your turn only when the whole task is done, or when you genuinely need a decision only the user can make (then ask a direct question). "+
			"Never stop mid-task with only a progress note.")
	systemMsgs := []any{map[string]any{
		"role":    "system",
		"content": strings.Join(systemParts, "\n\n"),
	}}
	// Append the actual conversation.
	//
	// Multimodal shape (matches the desktop client): an image-bearing user
	// message moves ALL of its content into a "contents" array, leaves "content"
	// as an EMPTY string, and orders the parts image-first then text. Filling
	// "content" as well, or sending text before the image, makes the gateway
	// treat the message as empty.
	for _, m := range req.Messages {
		// System content was folded into the merged system message above;
		// appending it again as a conversation entry would duplicate it.
		if m.Role == "system" {
			continue
		}
		if len(m.Images) == 0 {
			systemMsgs = append(systemMsgs, map[string]any{
				"role":    m.Role,
				"content": m.Content,
			})
			continue
		}
		parts := make([]map[string]any, 0, len(m.Images)+1)
		for _, img := range m.Images {
			url := img.URL
			if url == "" && img.Base64 != "" {
				mt := img.MediaType
				if mt == "" {
					mt = "image/png"
				}
				url = "data:" + mt + ";base64," + img.Base64
			}
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": url},
			})
		}
		if m.Content != "" {
			parts = append(parts, map[string]any{"type": "text", "text": m.Content})
		}
		systemMsgs = append(systemMsgs, map[string]any{
			"role":                    m.Role,
			"content":                 "",
			"contents":                parts,
			"response_meta":           map[string]any{"id": "", "usage": map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
			"reasoning_content_signature": "",
		})
	}
	base["messages"] = systemMsgs

	// Tools and tool_choice: pass the caller's own definitions through. The
	// template's fixed Qoder tool list is only a fallback for callers that send
	// none (e.g. a plain OpenAI chat client).
	if len(req.Tools) > 0 {
		base["tools"] = req.Tools
	}
	if len(req.ToolChoice) > 0 {
		base["tool_choice"] = req.ToolChoice
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
