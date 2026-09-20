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

// openAIMessage is one message in the chat completion format. Content accepts
// both shapes seen in practice:
//
//	"content": "text"                                  (OpenAI chat completions)
//	"content": [{"type":"text","text":"…"}]            (Anthropic messages)
//
// The Anthropic form is what the CPA /v1/messages front end forwards, so a
// plain string field here makes every such request fail with
// "cannot unmarshal array into Go struct field".
type openAIMessage struct {
	Role    string
	Content string
}

// UnmarshalJSON decodes either a bare string or an array of typed content parts,
// concatenating the text parts. Unknown part types are skipped rather than
// erroring, so an image or tool part does not sink an otherwise valid prompt.
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
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return fmt.Errorf("message content must be a string or an array of parts: %w", err)
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Text != "" {
			sb.WriteString(p.Text)
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
func extractLatestUserPrompt(messages []openAIMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}

// buildQwenBody renders the upstream agent_chat_generation body for one request.
// modelKey is the upstream key (already mapped via cpaToUpstreamKey).
func buildQwenBody(req *openAIRequest, modelKey, userType string) ([]byte, error) {
	var base map[string]any
	if err := json.Unmarshal(basepromptJSON, &base); err != nil {
		return nil, fmt.Errorf("baseprompt decode: %w", err)
	}

	prompt := extractLatestUserPrompt(req.Messages)
	if prompt == "" {
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
	// Append the actual conversation
	for _, m := range req.Messages {
		systemMsgs = append(systemMsgs, map[string]any{
			"role":    m.Role,
			"content": m.Content,
		})
	}
	base["messages"] = systemMsgs

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
