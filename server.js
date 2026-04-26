const express = require("express");
const axios = require("axios");
const { createStreamState, translateClaudeEvent, createChunk, buildFinishReasonChunk } = require("./stream-translator");

const app = express();
app.use(express.json({ limit: "250mb" }));

// AZURE_ENDPOINT is the bare Foundry host for Anthropic models, e.g.
//   https://<resource>.services.ai.azure.com
// The server appends /anthropic/v1/messages automatically. The legacy full-URL
// form ending in /anthropic/v1/messages is still accepted for backward compat.
const ANTHROPIC_MESSAGES_PATH = "/anthropic/v1/messages";

function stripTrailingSlashes(s) {
    return s ? s.replace(/\/+$/, "") : s;
}

function deriveAnthropicUrl(fullEndpoint) {
    if (!fullEndpoint) return undefined;
    const trimmed = stripTrailingSlashes(fullEndpoint);
    if (/\/anthropic\/v\d+\/messages$/i.test(trimmed)) return trimmed;
    try {
        const u = new URL(trimmed);
        return `${u.protocol}//${u.host}${ANTHROPIC_MESSAGES_PATH}`;
    } catch {
        return `${trimmed.replace(/^(https?:\/\/[^/]+).*/i, "$1")}${ANTHROPIC_MESSAGES_PATH}`;
    }
}

const CONFIG = {
    AZURE_ENDPOINT: deriveAnthropicUrl(process.env.AZURE_ENDPOINT),
    AZURE_API_KEY: process.env.AZURE_API_KEY,
    SERVICE_API_KEY: process.env.SERVICE_API_KEY,
    PORT: process.env.PORT || 8080,
    ANTHROPIC_VERSION: "2023-06-01",
};

// Azure OpenAI lives on a DIFFERENT host than Anthropic in Foundry:
//   Anthropic: <resource>.services.ai.azure.com
//   OpenAI:    <resource>.cognitiveservices.azure.com
// So AZURE_OPENAI_ENDPOINT is a distinct env var with no auto-derivation.
//
// Two URL styles are supported:
//  1) v1 (recommended): endpoint ends with `/openai/v1` (or `/openai/v1/`).
//     Request URL → `<endpoint>/chat/completions?api-version=preview`
//     Example:    https://<res>.cognitiveservices.azure.com/openai/v1
//  2) Legacy: bare host or any other path.
//     Request URL → `<host>/openai/deployments/<deployment>/chat/completions?api-version=<AZURE_OPENAI_API_VERSION>`
//     Example:    https://<res>.cognitiveservices.azure.com
function resolveGptUrlShape(endpoint) {
    if (!endpoint) return null;
    const base = stripTrailingSlashes(endpoint);
    if (/\/openai\/v1$/i.test(base)) return { style: "v1", base };
    // Tolerate a trailing "/openai" on the host
    const host = base.replace(/\/openai$/i, "");
    try {
        const u = new URL(host);
        return { style: "deployment", base: `${u.protocol}//${u.host}` };
    } catch {
        return { style: "deployment", base: host.replace(/^(https?:\/\/[^/]+).*/i, "$1") };
    }
}

function buildGptUrl(shape, deployment, apiVersion) {
    if (shape.style === "v1") {
        return `${shape.base}/chat/completions?api-version=preview`;
    }
    return `${shape.base}/openai/deployments/${encodeURIComponent(deployment)}/chat/completions?api-version=${encodeURIComponent(apiVersion)}`;
}

// Responses API lives at `<host>/openai/responses?api-version=...`. GPT‑5.x
// reasoning models (including gpt-5.4, gpt-5-pro, gpt-5-codex) are native to
// this API; chat/completions is a best-effort legacy shim that drops
// reasoning/summary/tool-call events. Use this shape when API_MODE=responses.
function buildGptResponsesUrl(shape, apiVersion) {
    if (!shape) return null;
    return `${shape.base.replace(/\/openai\/v1$/i, "").replace(/\/openai$/i, "")}/openai/responses?api-version=${encodeURIComponent(apiVersion)}`;
}

const GPT_CONFIG = (() => {
    const raw = process.env.AZURE_OPENAI_ENDPOINT;
    const shape = resolveGptUrlShape(raw);
    const mode = (process.env.AZURE_GPT_API_MODE || "chat").toLowerCase();
    return {
        RAW_ENDPOINT: raw,
        SHAPE: shape,
        API_KEY: process.env.AZURE_OPENAI_API_KEY || process.env.AZURE_API_KEY,
        API_VERSION: process.env.AZURE_OPENAI_API_VERSION || "2025-04-01-preview",
        DEPLOYMENT: process.env.AZURE_GPT_DEPLOYMENT || "gpt-5.4",
        DEFAULT_EFFORT: (process.env.GPT_REASONING_EFFORT || "medium").toLowerCase(),
        // "chat" = /openai/v1/chat/completions (legacy). "responses" = /openai/responses
        // (required for gpt-5.4 / gpt-5-pro / gpt-5-codex — see Cursor-Azure-GPT-5).
        API_MODE: mode === "responses" ? "responses" : "chat",
        SUMMARY_LEVEL: (process.env.AZURE_GPT_SUMMARY_LEVEL || "detailed").toLowerCase(),
        VERBOSITY_LEVEL: (process.env.AZURE_GPT_VERBOSITY_LEVEL || "medium").toLowerCase(),
        TRUNCATION: (process.env.AZURE_GPT_TRUNCATION || "disabled").toLowerCase(),
        // Cursor renders content inside <think>...</think> as reasoning. Turn off
        // if a client renders literal tags (some strict chat UIs).
        EMIT_THINK_TAGS: (process.env.AZURE_GPT_EMIT_THINK_TAGS || "true").toLowerCase() !== "false",
        // Kept for startup-log back-compat
        ENDPOINT: shape ? shape.base : undefined,
    };
})();

// ─── Model Routing ───────────────────────────────────────────────────────────

// Claude deployment names are configurable via env so users can pin a specific
// version (e.g. AZURE_CLAUDE_DEPLOYMENT_NAME=claude-opus-4-7). AZURE_CLAUDE_DEPLOYMENT_NAME
// acts as the overall default; per-family vars override when set.
const CLAUDE_DEFAULT = process.env.AZURE_CLAUDE_DEPLOYMENT_NAME || "claude-opus-4-6";
const CLAUDE_DIRECT_DEPLOYMENTS = (process.env.AZURE_CLAUDE_DEPLOYMENTS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

const MODEL_MAP = {
    "opus":   process.env.AZURE_CLAUDE_OPUS_DEPLOYMENT   || CLAUDE_DEFAULT,
    "sonnet": process.env.AZURE_CLAUDE_SONNET_DEPLOYMENT || CLAUDE_DEFAULT,
    "haiku":  process.env.AZURE_CLAUDE_HAIKU_DEPLOYMENT  || CLAUDE_DEFAULT,
};

const DEFAULT_DEPLOYMENT = CLAUDE_DEFAULT;

const CLAUDE_ALIAS_MAP = {
    "claudeopus-4-6": "claude-opus-4-6",
    "claudeopus-4-7": "claude-opus-4-7",
    "opus46": "claude-opus-4-6",
    "opus47": "claude-opus-4-7",
};

// Reasoning-effort suffixes understood for gpt-5.4 (longest first for greedy match).
const GPT_EFFORT_SUFFIXES = ["minimal", "medium", "high", "low"];

// Cursor-facing gpt-5.4 model ids and the reasoning_effort each resolves to.
// The bare `gpt-5.4` id falls back to GPT_CONFIG.DEFAULT_EFFORT (env-controlled).
const GPT_MODEL_MAP = {
    "gpt-5.4":          GPT_CONFIG.DEFAULT_EFFORT,
    "gpt-5.4-minimal":  "minimal",
    "gpt-5.4-low":      "low",
    "gpt-5.4-medium":   "medium",
    "gpt-5.4-high":     "high",
};

function isGptModel(cursorModel) {
    if (!cursorModel) return false;
    return cursorModel.toLowerCase().startsWith("gpt-");
}

function extractGptReasoningEffort(cursorModel) {
    if (!cursorModel) return GPT_CONFIG.DEFAULT_EFFORT;
    const lower = cursorModel.toLowerCase();
    for (const level of GPT_EFFORT_SUFFIXES) {
        if (lower.endsWith(`-${level}`)) return level;
    }
    return GPT_CONFIG.DEFAULT_EFFORT;
}

function resolveDeployment(cursorModel) {
    if (!cursorModel) return DEFAULT_DEPLOYMENT;
    const lower = cursorModel.toLowerCase();

    if (CLAUDE_ALIAS_MAP[lower]) {
        return CLAUDE_ALIAS_MAP[lower];
    }

    const compactClaudeMatch = lower.match(/^claude(opus|sonnet|haiku)-(.+)$/);
    if (compactClaudeMatch) {
        return `claude-${compactClaudeMatch[1]}-${compactClaudeMatch[2]}`;
    }

    // If Cursor sends a concrete Azure Claude deployment name, use it directly.
    // This lets "claude-opus-4-6" and "claude-opus-4-7" target different
    // deployments without being rewritten through the opus family default.
    if (/^claude-[a-z0-9]+(?:-[a-z0-9]+)*-\d+(?:[.-]\d+)*$/i.test(cursorModel)) {
        return cursorModel;
    }

    // Direct match in MODEL_MAP families / aliases
    for (const [family, deployment] of Object.entries(MODEL_MAP)) {
        if (lower.includes(family)) return deployment;
    }

    // Handle Cursor naming patterns like "claude-4-6", "claude-sonnet-4-6", "claude4sonnet", etc.
    // Also catch versioned names like "claude-3-5-sonnet", "claude-3.5-haiku"
    if (lower.includes("claude")) {
        // Check for specific model tiers in any position
        if (lower.includes("opus")) return MODEL_MAP["opus"];
        if (lower.includes("haiku")) return MODEL_MAP["haiku"];
        // Default Claude requests to sonnet (most capable general-purpose)
        return MODEL_MAP["sonnet"];
    }

    return DEFAULT_DEPLOYMENT;
}

// ─── Middleware ───────────────────────────────────────────────────────────────

app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
    if (req.method === "OPTIONS") return res.sendStatus(200);
    next();
});

app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

function requireAuth(req, res, next) {
    if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") return next();

    if (!CONFIG.SERVICE_API_KEY) {
        return res.status(500).json({ error: { message: "SERVICE_API_KEY not configured", type: "configuration_error" } });
    }

    const authHeader = req.headers.authorization;
    if (!authHeader) {
        return res.status(401).json({
            error: {
                message: "Missing Authorization header. Set OpenAI API Key in Cursor to match SERVICE_API_KEY in .env",
                type: "authentication_error",
            },
        });
    }

    const token = authHeader.startsWith("Bearer ") ? authHeader.substring(7) : authHeader;
    if (token !== CONFIG.SERVICE_API_KEY) {
        return res.status(401).json({
            error: {
                message: "Invalid API key. Cursor's OpenAI API Key must match SERVICE_API_KEY in .env",
                type: "authentication_error",
            },
        });
    }

    next();
}

// ─── OpenAI → Anthropic Request Translation ──────────────────────────────────

function convertMessagesToAnthropic(openaiMessages) {
    let systemParts = [];
    const anthropicMessages = [];
    const seenToolIds = new Set();

    for (const msg of openaiMessages) {
        if (msg.role === "system" || msg.role === "developer") {
            const text = typeof msg.content === "string"
                ? msg.content
                : Array.isArray(msg.content)
                    ? msg.content.map((c) => c.text || "").join("\n")
                    : String(msg.content || "");
            if (text) systemParts.push(text);
            continue;
        }

        if (msg.role === "assistant") {
            const contentBlocks = [];

            // Issue 1: Thinking blocks must round-trip through conversation history.
            // Anthropic requires them FIRST in assistant content (factory.py:2443-2447).
            if (msg.thinking_blocks && Array.isArray(msg.thinking_blocks)) {
                for (const tb of msg.thinking_blocks) {
                    if (tb.type === "thinking" && tb.thinking) {
                        contentBlocks.push({ type: "thinking", thinking: tb.thinking, signature: tb.signature || "" });
                    } else if (tb.type === "redacted_thinking") {
                        contentBlocks.push({ type: "redacted_thinking", data: tb.data || "" });
                    }
                }
            }
            // Also handle reasoning_content (alternative format some clients use)
            if (msg.reasoning_content && typeof msg.reasoning_content === "string" && msg.reasoning_content.length > 0) {
                if (contentBlocks.length === 0 || contentBlocks[0].type !== "thinking") {
                    contentBlocks.unshift({ type: "thinking", thinking: msg.reasoning_content, signature: msg.reasoning_signature || "" });
                }
            }

            if (msg.content) {
                if (typeof msg.content === "string") {
                    if (msg.content.length > 0) {
                        contentBlocks.push({ type: "text", text: msg.content });
                    }
                } else if (Array.isArray(msg.content)) {
                    for (const part of msg.content) {
                        if (part.type === "text" && part.text && part.text.length > 0) {
                            contentBlocks.push({ type: "text", text: part.text });
                        } else if (part.type === "thinking" && part.thinking) {
                            contentBlocks.push({ type: "thinking", thinking: part.thinking, signature: part.signature || "" });
                        } else if (part.type === "redacted_thinking") {
                            contentBlocks.push({ type: "redacted_thinking", data: part.data || "" });
                        } else if (part.type === "tool_use" && part.id && part.name) {
                            if (!seenToolIds.has(part.id)) {
                                seenToolIds.add(part.id);
                                contentBlocks.push({
                                    type: "tool_use",
                                    id: part.id,
                                    name: part.name,
                                    input: part.input || {},
                                });
                            }
                        }
                    }
                }
            }


            // Standard OpenAI tool_calls field
            if (msg.tool_calls && msg.tool_calls.length > 0) {
                for (const tc of msg.tool_calls) {
                    if (seenToolIds.has(tc.id)) continue;
                    seenToolIds.add(tc.id);

                    let input = {};
                    try { input = JSON.parse(tc.function.arguments); } catch { input = {}; }
                    contentBlocks.push({
                        type: "tool_use",
                        id: tc.id,
                        name: tc.function.name,
                        input,
                    });
                }
            }

            // Legacy OpenAI function_call field
            if (msg.function_call && msg.function_call.name) {
                const fcId = msg.function_call.id || `fc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
                if (!seenToolIds.has(fcId)) {
                    seenToolIds.add(fcId);
                    let input = {};
                    try { input = JSON.parse(msg.function_call.arguments || "{}"); } catch { input = {}; }
                    contentBlocks.push({
                        type: "tool_use",
                        id: fcId,
                        name: msg.function_call.name,
                        input,
                    });
                }
            }

            // Issue 4: Never skip assistant messages — breaks required role alternation.
            // If content is empty (e.g. content:null + no tool_calls), push a placeholder.
            if (contentBlocks.length === 0) {
                contentBlocks.push({ type: "text", text: "." });
            }
            anthropicMessages.push({ role: "assistant", content: contentBlocks });
            continue;
        }

        if (msg.role === "tool") {
            // Issue 5: Validate tool_call_id presence
            const toolUseId = msg.tool_call_id;
            if (!toolUseId) {
                console.warn(`[CONVERT] Tool result message missing tool_call_id, generating placeholder`);
            }
            const toolResult = {
                type: "tool_result",
                tool_use_id: toolUseId || `toolu_placeholder_${Date.now()}`,
                content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
            };

            const prev = anthropicMessages[anthropicMessages.length - 1];
            if (prev && prev.role === "user" && Array.isArray(prev.content) && prev.content[0]?.type === "tool_result") {
                prev.content.push(toolResult);
            } else {
                anthropicMessages.push({ role: "user", content: [toolResult] });
            }
            continue;
        }

        // Legacy OpenAI function result format
        if (msg.role === "function") {
            let toolUseId = msg.tool_call_id;
            if (!toolUseId && msg.name) {
                // Match against the preceding assistant's tool_use blocks by function name
                const prevAst = anthropicMessages.length > 0 ? anthropicMessages[anthropicMessages.length - 1] : null;
                if (!prevAst || prevAst.role !== "assistant") {
                    // Check one level back (there might be a user tool_result message in between)
                    for (let j = anthropicMessages.length - 1; j >= 0; j--) {
                        if (anthropicMessages[j].role === "assistant") { const found = anthropicMessages[j]; if (Array.isArray(found.content)) { const match = found.content.find(b => b.type === "tool_use" && b.name === msg.name); if (match) { toolUseId = match.id; } } break; }
                    }
                } else if (Array.isArray(prevAst.content)) {
                    const match = prevAst.content.find(b => b.type === "tool_use" && b.name === msg.name);
                    if (match) toolUseId = match.id;
                }
            }
            toolUseId = toolUseId || `fc_result_${Date.now()}`;
            const toolResult = {
                type: "tool_result",
                tool_use_id: toolUseId,
                content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
            };
            const prev = anthropicMessages[anthropicMessages.length - 1];
            if (prev && prev.role === "user" && Array.isArray(prev.content) && prev.content[0]?.type === "tool_result") {
                prev.content.push(toolResult);
            } else {
                anthropicMessages.push({ role: "user", content: [toolResult] });
            }
            continue;
        }

        if (msg.role === "user") {
            let content;
            if (typeof msg.content === "string") {
                content = msg.content;
            } else if (Array.isArray(msg.content)) {
                content = msg.content.map((part) => {
                    if (part.type === "text") return { type: "text", text: part.text };
                    if (part.type === "tool_result" && part.tool_use_id) {
                        return { type: "tool_result", tool_use_id: part.tool_use_id, content: part.content || "" };
                    }
                    if (part.type === "image_url") {
                        const url = part.image_url?.url || "";
                        if (url.startsWith("data:")) {
                            const match = url.match(/^data:(image\/[^;]+);base64,(.+)$/);
                            if (match) {
                                return { type: "image", source: { type: "base64", media_type: match[1], data: match[2] } };
                            }
                        }
                        return { type: "text", text: `[Image: ${url}]` };
                    }
                    return { type: "text", text: JSON.stringify(part) };
                });
            } else {
                content = String(msg.content || "");
            }
            anthropicMessages.push({ role: "user", content });
            continue;
        }

        // Catch-all: warn about unhandled roles so they don't silently vanish
        console.warn(`[CONVERT] Unhandled message role="${msg.role}", converting to user text`)
        const fallbackText = typeof msg.content === "string" ? msg.content
            : msg.content ? JSON.stringify(msg.content) : ".";
        anthropicMessages.push({ role: "user", content: fallbackText || "." });
    }

    // Deduplicate tool_result blocks by tool_use_id within each user message
    // (PR #23104: session replay can produce duplicate tool_result for the same tool_use_id)
    for (const msg of anthropicMessages) {
        if (msg.role === "user" && Array.isArray(msg.content)) {
            const seenResultIds = new Map();
            const toRemove = new Set();
            for (let i = 0; i < msg.content.length; i++) {
                const block = msg.content[i];
                if (block.type === "tool_result" && block.tool_use_id) {
                    if (seenResultIds.has(block.tool_use_id)) {
                        toRemove.add(seenResultIds.get(block.tool_use_id));
                        console.warn(`[CONVERT] Dedup tool_result for ${block.tool_use_id} (keeping last)`);
                    }
                    seenResultIds.set(block.tool_use_id, i);
                }
            }
            if (toRemove.size > 0) {
                msg.content = msg.content.filter((_, i) => !toRemove.has(i));
            }
        }
    }

    // Anthropic requires alternating user/assistant — merge consecutive same-role messages
    const merged = [];
    for (const msg of anthropicMessages) {
        const prev = merged[merged.length - 1];
        if (prev && prev.role === msg.role) {
            const prevContent = Array.isArray(prev.content)
                ? prev.content
                : [{ type: "text", text: String(prev.content) }];
            const currContent = Array.isArray(msg.content)
                ? msg.content
                : [{ type: "text", text: String(msg.content) }];
            prev.content = [...prevContent, ...currContent];
        } else {
            merged.push({ ...msg });
        }
    }

    if (merged.length > 0 && merged[0].role !== "user") {
        merged.unshift({ role: "user", content: "." });
    }

    // Orphan repair: ensure every tool_use has a tool_result and vice versa
    // (LiteLLM sanitize_messages_for_tool_calling Cases A-C)
    const allToolUseIds = new Set();
    const allToolResultIds = new Set();
    for (const msg of merged) {
        if (!Array.isArray(msg.content)) continue;
        for (const block of msg.content) {
            if (block.type === "tool_use") allToolUseIds.add(block.id);
            if (block.type === "tool_result") allToolResultIds.add(block.tool_use_id);
        }
    }
    // Case A: tool_use without tool_result → inject dummy result in the next user message
    for (const useId of allToolUseIds) {
        if (!allToolResultIds.has(useId)) {
            console.warn(`[CONVERT] Orphan tool_use ${useId} — injecting dummy tool_result`);
            for (let i = 0; i < merged.length; i++) {
                if (!Array.isArray(merged[i].content)) continue;
                const hasThisUse = merged[i].content.some(b => b.type === "tool_use" && b.id === useId);
                if (hasThisUse) {
                    const nextUser = merged[i + 1];
                    const dummy = { type: "tool_result", tool_use_id: useId, content: "[No result captured]" };
                    if (nextUser && nextUser.role === "user") {
                        const arr = Array.isArray(nextUser.content)
                            ? nextUser.content : [{ type: "text", text: String(nextUser.content) }];
                        nextUser.content = [dummy, ...arr];
                    } else {
                        merged.splice(i + 1, 0, { role: "user", content: [dummy] });
                    }
                    break;
                }
            }
        }
    }
    // Case B: tool_result without tool_use → remove the orphaned result
    for (const msg of merged) {
        if (!Array.isArray(msg.content)) continue;
        msg.content = msg.content.filter(block => {
            if (block.type === "tool_result" && !allToolUseIds.has(block.tool_use_id)) {
                console.warn(`[CONVERT] Orphan tool_result ${block.tool_use_id} — removing`);
                return false;
            }
            return true;
        });
        if (msg.content.length === 0) {
            msg.content = [{ type: "text", text: "." }];
        }
    }

    return { system: systemParts.length > 0 ? systemParts.join("\n\n") : undefined, messages: merged };
}

function convertToolsToAnthropic(openaiTools) {
    if (!openaiTools || !Array.isArray(openaiTools) || openaiTools.length === 0) return undefined;
    return openaiTools.map((tool) => {
        if (tool.type === "function") {
            return {
                name: tool.function.name,
                description: tool.function.description || "",
                input_schema: tool.function.parameters || { type: "object", properties: {} },
            };
        }
        return tool;
    });
}

function convertToolChoiceToAnthropic(openaiToolChoice, parallelToolCalls) {
    let result;
    if (!openaiToolChoice || openaiToolChoice === "auto") {
        result = { type: "auto" };
    } else if (openaiToolChoice === "required") {
        result = { type: "any" };
    } else if (openaiToolChoice === "none") {
        return undefined;
    } else if (typeof openaiToolChoice === "object" && openaiToolChoice.type === "function") {
        result = { type: "tool", name: openaiToolChoice.function.name };
    } else if (typeof openaiToolChoice === "object" && openaiToolChoice.type) {
        result = { type: openaiToolChoice.type };
        if (openaiToolChoice.type === "tool" && openaiToolChoice.name) result.name = openaiToolChoice.name;
    } else {
        return undefined;
    }
    // Anthropic uses disable_parallel_tool_use (inverse of OpenAI's parallel_tool_calls)
    if (parallelToolCalls === false) {
        result.disable_parallel_tool_use = true;
    } else if (parallelToolCalls === true) {
        result.disable_parallel_tool_use = false;
    }
    return result;
}

function shouldEnableThinking(modelName) {
    if (!modelName) return false;
    const lower = modelName.toLowerCase();
    return lower.includes("thinking") || lower.includes("think");
}

// Family-based limits so custom deployment names (e.g. claude-opus-4-7) still resolve correctly.
const MODEL_MAX_OUTPUT_BY_FAMILY = {
    opus: 32000,
    sonnet: 64000,
    haiku: 8192,
};
// With thinking enabled, 128K is natively supported (no beta header needed)
const THINKING_MAX_OUTPUT = 128000;
// Azure AI supported beta headers (from LiteLLM's anthropic_beta_headers_config.json).
// NOT supported on Azure: output-128k, token-efficient-tools, fine-grained-tool-streaming,
// compact, fast-mode. Sending unsupported headers can cause silent failures.
const AZURE_SUPPORTED_BETA_FLAGS = [];
const MIN_OUTPUT_TOKENS = 16384;
// Adaptive thinking (Opus 4.6+) — no budget_tokens needed, the model decides autonomously.
// Effort level controls how much thinking the model does: "high" (default), "medium", "low".
const THINKING_EFFORT = (process.env.THINKING_EFFORT || "high").toLowerCase();

function resolveModelMax(deployment) {
    if (!deployment) return 64000;
    const lower = deployment.toLowerCase();
    for (const [family, max] of Object.entries(MODEL_MAX_OUTPUT_BY_FAMILY)) {
        if (lower.includes(family)) return max;
    }
    return 64000;
}

function resolveMaxTokens(openaiMaxTokens, deployment, thinkingEnabled) {
    if (thinkingEnabled) return THINKING_MAX_OUTPUT;
    const modelMax = resolveModelMax(deployment);
    if (!openaiMaxTokens || openaiMaxTokens < MIN_OUTPUT_TOKENS) {
        return Math.min(modelMax, Math.max(MIN_OUTPUT_TOKENS, modelMax));
    }
    return Math.min(openaiMaxTokens, modelMax);
}

function buildAnthropicRequest(openaiBody) {
    const { system, messages } = convertMessagesToAnthropic(openaiBody.messages || []);
    const thinkingRequested = shouldEnableThinking(openaiBody.model);
    const deployment = resolveDeployment(openaiBody.model);
    const hasTools = openaiBody.tools && openaiBody.tools.length > 0;
    const toolChoiceIsNone = openaiBody.tool_choice === "none";
    const effectiveHasTools = hasTools && !toolChoiceIsNone;

    const thinkingEnabled = thinkingRequested;

    const maxTokensFromClient = openaiBody.max_tokens || openaiBody.max_completion_tokens;
    const anthropicReq = {
        model: deployment,
        messages,
        max_tokens: resolveMaxTokens(maxTokensFromClient, deployment, thinkingEnabled),
    };

    if (system) anthropicReq.system = system;
    if (openaiBody.stream !== undefined) anthropicReq.stream = openaiBody.stream;
    if (openaiBody.stop) anthropicReq.stop_sequences = Array.isArray(openaiBody.stop) ? openaiBody.stop : [openaiBody.stop];

    if (thinkingEnabled) {
        anthropicReq.thinking = { type: "adaptive" };
        anthropicReq.output_config = { effort: THINKING_EFFORT };
    } else {
        if (openaiBody.temperature !== undefined) anthropicReq.temperature = openaiBody.temperature;
        if (openaiBody.top_p !== undefined) anthropicReq.top_p = openaiBody.top_p;
    }

    if (effectiveHasTools) {
        anthropicReq.tools = convertToolsToAnthropic(openaiBody.tools);
        const toolChoice = convertToolChoiceToAnthropic(openaiBody.tool_choice, openaiBody.parallel_tool_calls);
        anthropicReq.tool_choice = toolChoice || { type: "auto" };

    }

    return anthropicReq;
}

// ─── Anthropic → OpenAI Response Translation ─────────────────────────────────

function anthropicStopToOpenai(stopReason) {
    switch (stopReason) {
        case "end_turn": return "stop";
        case "tool_use": return "tool_calls";
        case "max_tokens": return "length";
        case "stop_sequence": return "stop";
        default: return "stop";
    }
}

function convertAnthropicResponseToOpenai(anthropicResp, requestModel) {
    const textParts = [];
    const toolCalls = [];

    for (const block of anthropicResp.content || []) {
        if (block.type === "text") {
            textParts.push(block.text);
        } else if (block.type === "tool_use") {
            toolCalls.push({
                id: block.id,
                type: "function",
                function: {
                    name: block.name,
                    arguments: JSON.stringify(block.input),
                },
            });
        }
    }

    const message = { role: "assistant", content: textParts.length > 0 ? textParts.join("") : null };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;

    let finishReason = anthropicStopToOpenai(anthropicResp.stop_reason);
    // Claude sometimes returns "end_turn" even when tool_use blocks are present;
    // Cursor needs "tool_calls" to trigger tool execution
    if (finishReason === "stop" && toolCalls.length > 0) {
        finishReason = "tool_calls";
    }

    return {
        id: anthropicResp.id || "chatcmpl-" + Date.now(),
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: requestModel || DEFAULT_DEPLOYMENT,
        choices: [{ index: 0, message, finish_reason: finishReason }],
        usage: {
            prompt_tokens: anthropicResp.usage?.input_tokens || 0,
            completion_tokens: anthropicResp.usage?.output_tokens || 0,
            total_tokens: (anthropicResp.usage?.input_tokens || 0) + (anthropicResp.usage?.output_tokens || 0),
        },
    };
}

// ─── Streaming: Anthropic SSE → OpenAI SSE ───────────────────────────────────

function writeChunk(res, chunk) {
    try {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    } catch {
        // Client disconnected
    }
}

const LOG_TOOL_CALLS = (process.env.LOG_TOOL_CALLS || "").toLowerCase() === "true";
const LOG_MESSAGES = (process.env.LOG_MESSAGES || "").toLowerCase() === "true";
const MAX_STREAM_DURATION_MS = 10 * 60 * 1000; // 10 minutes

// Attempts to close truncated JSON by appending missing quotes/brackets/braces.
// Returns the suffix to append, or null if the JSON is already valid or unrepairable.
// Needed because Anthropic streams can be cut off mid-tool-argument
// (see github.com/anthropics/anthropic-sdk-typescript/issues/842).
function repairTruncatedJSON(str) {
    if (!str) return "{}";
    try { JSON.parse(str); return null; } catch {}

    let inStr = false, esc = false;
    const stack = [];
    for (const ch of str) {
        if (esc) { esc = false; continue; }
        if (ch === '\\' && inStr) { esc = true; continue; }
        if (ch === '"') { inStr = !inStr; continue; }
        if (inStr) continue;
        if (ch === '{') stack.push('}');
        else if (ch === '[') stack.push(']');
        else if ((ch === '}' || ch === ']') && stack.length) stack.pop();
    }
    const close = stack.reverse().join('');
    const candidates = inStr
        ? [`"${close}`, `":null${close}`]
        : [close, `null${close}`, `"_":null${close}`];
    for (const fix of candidates) {
        try { JSON.parse(str + fix); return fix; } catch {}
    }
    return null;
}

function handleAnthropicStream(axiosResponse, res, requestModel, abortController, headersAlreadySent = false) {
    if (!headersAlreadySent) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();
    }

    const state = createStreamState(requestModel || DEFAULT_DEPLOYMENT);
    const streamStart = Date.now();
    let buffer = "";
    let streamEnded = false;
    let streamCompleted = false;
    let lastActivityTime = Date.now();

    const heartbeatInterval = setInterval(() => {
        if (streamEnded || res.writableEnded) return;

        const elapsed = Date.now() - streamStart;
        if (elapsed > MAX_STREAM_DURATION_MS) {
            console.log(`[STREAM] Max duration reached (${elapsed}ms), closing`);
            writeSSEError(res, "Stream exceeded maximum duration", "timeout_error");
            cleanup();
            return;
        }

        const idleTime = Date.now() - lastActivityTime;
        if (idleTime > 120000) {
            console.log(`[STREAM] Idle for ${idleTime}ms, closing`);
            cleanup();
            return;
        }

        try { res.write(": heartbeat\n\n"); } catch { cleanup(); }
    }, 15000);

    function cleanup() {
        if (streamEnded) return;
        streamEnded = true;
        clearInterval(heartbeatInterval);
        if (!streamCompleted && abortController) {
            try { abortController.abort(); } catch {}
        }
        if (!res.writableEnded) {
            try { res.end(); } catch {}
        }
    }

    res.on("close", () => {
        if (!streamEnded) {
            console.log("[STREAM] Client disconnected, aborting upstream");
            cleanup();
        }
    });

    axiosResponse.data.on("data", (chunk) => {
        if (streamEnded) return;
        lastActivityTime = Date.now();

        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop();

        let currentEventType = null;
        for (const line of lines) {
            const trimmed = line.trim();

            if (trimmed.startsWith("event:")) {
                currentEventType = trimmed.slice(6).trim();
                continue;
            }
            if (!trimmed.startsWith("data:")) {
                if (trimmed === "") currentEventType = null;
                continue;
            }

            const data = trimmed.slice(5).trim();
            if (data === "[DONE]") {
                currentEventType = null;
                continue;
            }

            let event;
            try { event = JSON.parse(data); } catch { currentEventType = null; continue; }

            if (currentEventType) {
                event.type = currentEventType;
                currentEventType = null;
            }

            // Handle errors embedded in the stream
            if (event.type === "error") {
                console.error(`[STREAM] Error event:`, event.error?.message || JSON.stringify(event));
                if (!res.writableEnded) {
                    writeSSEError(res, event.error?.message || "Stream error from upstream");
                }
                cleanup();
                return;
            }

            if (LOG_TOOL_CALLS) {
                if (event.type === "content_block_start" && event.content_block?.type === "tool_use") {
                    console.log(`[DIAG] tool_use start: id=${event.content_block.id}, name=${event.content_block.name}, index=${event.index}`);
                }
                if (event.type === "content_block_stop" && state.toolCalls.has(event.index)) {
                    const tc = state.toolCalls.get(event.index);
                    console.log(`[DIAG] tool_use end: index=${tc.index}, name=${tc.function.name}, args_len=${tc.function.arguments.length}`);
                }
                if (event.type === "message_delta" && event.delta?.stop_reason) {
                    console.log(`[DIAG] message_delta stop_reason=${event.delta.stop_reason}`);
                }
            }

            const chunks = translateClaudeEvent(event, state);
            if (chunks) {
                for (const c of chunks) {
                    if (LOG_TOOL_CALLS && c.choices?.[0]?.finish_reason) {
                        console.log(`[DIAG] outgoing finish_reason=${c.choices[0].finish_reason}`);
                    }
                    writeChunk(res, c);
                }
            }

            if (event.type === "message_stop") {
                const outputTokens = state.usage?.output_tokens || 0;
                const inputTokens = state.usage?.input_tokens || 0;
                console.log(`[STREAM] Done: finish=${state.finishReason}, tools=${state.toolCalls.size}, usage=${JSON.stringify(state.usage || {})}`);
                if (state.finishReason === "stop" && outputTokens < 50 && state.toolCalls.size === 0) {
                    console.warn(`[WARN] Suspiciously short response: ${outputTokens} output tokens with end_turn and no tool calls (input_tokens=${inputTokens}). Model may have lost context.`);
                }
                streamCompleted = true;
                try { res.write("data: [DONE]\n\n"); } catch {}
                cleanup();
            }
        }
    });

    axiosResponse.data.on("end", () => {
        console.log("[STREAM] Upstream ended");

        // Flush remaining buffer — may contain complete SSE events without trailing \n
        if (buffer.trim()) {
            const remaining = buffer;
            buffer = "";
            for (const line of remaining.split("\n")) {
                const trimmed = line.trim();
                if (!trimmed.startsWith("data:")) continue;
                const data = trimmed.slice(5).trim();
                if (data === "[DONE]") continue;
                try {
                    const event = JSON.parse(data);
                    const chunks = translateClaudeEvent(event, state);
                    if (chunks) {
                        for (const c of chunks) writeChunk(res, c);
                    }
                    if (event.type === "message_stop") {
                        streamCompleted = true;
                        try { res.write("data: [DONE]\n\n"); } catch {}
                    }
                } catch {}
            }
        }

        if (!streamCompleted && !streamEnded && !res.writableEnded) {
            // Upstream closed without message_stop — known Anthropic streaming bug
            // that truncates tool call arguments mid-transmission
            // (see github.com/anthropics/anthropic-sdk-typescript/issues/842).
            if (state.toolCalls.size > 0) {
                for (const [, toolCall] of state.toolCalls) {
                    const repair = repairTruncatedJSON(toolCall.function.arguments);
                    if (repair) {
                        console.log(`[STREAM] Repairing truncated tool_call[${toolCall.index}] "${toolCall.function.name}" — appending: ${repair}`);
                        writeChunk(res, createChunk(state, {
                            tool_calls: [{
                                index: toolCall.index,
                                function: { arguments: repair },
                            }],
                        }));
                    }
                }
            }

            if (!state.finishReasonSent) {
                const finalChunk = buildFinishReasonChunk(state);
                writeChunk(res, finalChunk);
                state.finishReasonSent = true;
                console.log(`[STREAM] Synthesized finish_reason=${finalChunk.choices[0].finish_reason} (upstream ended without message_stop)`);
            }
            try { res.write("data: [DONE]\n\n"); } catch {}
        }
        cleanup();
    });

    axiosResponse.data.on("error", (error) => {
        console.error("[STREAM] Error:", error.message);
        if (!res.writableEnded) {
            writeSSEError(res, "Stream error: " + error.message);
        }
        cleanup();
    });
}

// ─── Chat Completions Handler ────────────────────────────────────────────────

function isModelValidationPing(body) {
    if (body.stream === true) return false;
    if (body.tools && body.tools.length > 0) return false;
    if (body.tool_choice) return false;
    const msgs = body.messages || [];
    if (msgs.length !== 1) return false;
    const content = msgs[0]?.content;
    if (typeof content !== "string") return false;
    if (content.length > 50) return false;
    return true;
}

function makeValidationResponse(model) {
    return {
        id: "chatcmpl-ping-" + Date.now(),
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
            index: 0,
            message: { role: "assistant", content: "Hello! I'm ready." },
            finish_reason: "stop",
        }],
        usage: { prompt_tokens: 1, completion_tokens: 3, total_tokens: 4 },
    };
}

function writeSSEError(res, message, type = "proxy_error") {
    const errorChunk = {
        id: "chatcmpl-error-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: "error",
        choices: [{ index: 0, delta: { content: `[Error: ${message}]` }, finish_reason: "stop" }],
    };
    try {
        res.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
    } catch {}
}

// ─── Azure OpenAI (GPT-5.x) Responses API adapter ────────────────────────────
//
// GPT‑5.x reasoning models (gpt-5.4, gpt-5-pro, gpt-5-codex, …) are native to
// Azure's Responses API (`/openai/responses`). Chat Completions on the v1
// preview endpoint often returns 200 but drops reasoning/tool events in a way
// Cursor renders as "Invalid API key". When AZURE_GPT_API_MODE=responses we
// translate Cursor's chat-completions request into Responses schema, stream
// from `/openai/responses`, and convert events back into chat.completion.chunk
// frames Cursor understands.
//
// This is a faithful port of the transform in Cursor-Azure-GPT-5
// (request_adapter.py / response_adapter.py).

function contentToText(content) {
    if (content == null) return "";
    if (typeof content === "string") return content;
    if (!Array.isArray(content) && typeof content === "object") {
        if (typeof content.text === "string") return content.text;
        if (typeof content.input_text === "string") return content.input_text;
        if (typeof content.output_text === "string") return content.output_text;
    }
    if (!Array.isArray(content)) return String(content);
    const parts = [];
    for (const part of content) {
        if (part && typeof part === "object") {
            if (part.type === "text" || part.type === "input_text" || part.type === "output_text") {
                parts.push(part.text || "");
            }
            else if (part.type === "image_url") parts.push("[image]");
            else if (typeof part.text === "string") parts.push(part.text);
            else parts.push(`[${part.type || "unknown"}]`);
        } else {
            parts.push(String(part));
        }
    }
    return parts.join("\n");
}

function contentToResponsesParts(content, role) {
    const textType = role === "assistant" ? "output_text" : "input_text";
    if (content == null) return [{ type: textType, text: "" }];
    if (typeof content === "string") return [{ type: textType, text: content }];
    if (!Array.isArray(content)) return [{ type: textType, text: contentToText(content) }];

    const parts = [];
    for (const part of content) {
        if (!part || typeof part !== "object") {
            parts.push({ type: textType, text: String(part) });
            continue;
        }
        if (part.type === "text" || part.type === "input_text" || part.type === "output_text") {
            parts.push({ type: textType, text: part.text || "" });
            continue;
        }
        if (part.type === "image_url" && role === "user") {
            const imageUrl = typeof part.image_url === "string" ? part.image_url : part.image_url?.url;
            if (imageUrl) {
                parts.push({ type: "input_image", image_url: imageUrl });
                continue;
            }
        }
        if (part.type === "input_image" && role === "user") {
            const imagePart = { type: "input_image" };
            if (part.image_url) imagePart.image_url = part.image_url;
            if (part.file_id) imagePart.file_id = part.file_id;
            if (part.detail) imagePart.detail = part.detail;
            parts.push(imagePart);
            continue;
        }
        parts.push({ type: textType, text: contentToText([part]) });
    }
    return parts.length ? parts : [{ type: textType, text: "" }];
}

function convertMessagesToResponsesInput(messages) {
    const instructionsParts = [];
    const input = [];
    for (const m of messages || []) {
        const role = m.role;
        if (role === "system" || role === "developer") {
            const t = contentToText(m.content);
            if (t) instructionsParts.push(t);
            continue;
        }
        if (role === "tool") {
            input.push({
                type: "function_call_output",
                output: contentToText(m.content),
                status: "completed",
                call_id: m.tool_call_id,
            });
            continue;
        }
        // user / assistant / anything else → text turn
        input.push({
            role: role || "user",
            content: contentToResponsesParts(m.content, role || "user"),
        });
        if (Array.isArray(m.tool_calls)) {
            for (const tc of m.tool_calls) {
                const fn = tc.function || {};
                input.push({
                    type: "function_call",
                    name: fn.name,
                    arguments: fn.arguments,
                    call_id: tc.id,
                });
            }
        }
    }
    return {
        instructions: instructionsParts.length ? instructionsParts.join("\n\n") : null,
        input: input.length ? input : null,
    };
}

function convertToolsToResponses(openaiTools) {
    if (!Array.isArray(openaiTools)) return [];
    const out = [];
    for (const tool of openaiTools) {
        const fn = tool && tool.function;
        if (!fn) continue;
        out.push({
            type: "function",
            name: fn.name,
            description: fn.description,
            parameters: fn.parameters,
            strict: false,
        });
    }
    return out;
}

function buildResponsesBody(openaiBody, effort) {
    const { instructions, input } = convertMessagesToResponsesInput(openaiBody.messages || []);

    // Azure rejects `input: null` with "expected a string, but got an object".
    // If we have no user/assistant/tool turns, fall back to a minimal stub so
    // the request is still well-formed. The instructions field still carries
    // any system/developer content.
    const safeInput = (Array.isArray(input) && input.length > 0)
        ? input
        : [{ role: "user", content: [{ type: "input_text", text: "" }] }];

    const body = {
        model: GPT_CONFIG.DEPLOYMENT,
        input: safeInput,
        tools: convertToolsToResponses(openaiBody.tools || []),
        tool_choice: openaiBody.tool_choice || "auto",
        // Always stream upstream — we buffer on our side when the client asked for non-streaming.
        stream: true,
        reasoning: { effort },
        store: false,
        stream_options: { include_obfuscation: false },
    };
    // Only include optional fields when they have a real value. Azure's validator
    // sometimes rejects nulls on fields that are documented as optional strings.
    if (instructions) body.instructions = instructions;
    if (typeof openaiBody.user === "string" && openaiBody.user.length > 0) {
        body.prompt_cache_key = openaiBody.user;
    }
    if (["auto", "detailed", "concise"].includes(GPT_CONFIG.SUMMARY_LEVEL)) {
        body.reasoning.summary = GPT_CONFIG.SUMMARY_LEVEL;
    }
    if (["low", "high"].includes(GPT_CONFIG.VERBOSITY_LEVEL)) {
        body.text = { verbosity: GPT_CONFIG.VERBOSITY_LEVEL };
    }
    if (GPT_CONFIG.TRUNCATION === "auto") body.truncation = "auto";
    return body;
}

function createResponsesStreamState(modelId) {
    return {
        chatId: "chatcmpl-" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36),
        model: modelId,
        thinking: false,
        toolCalls: 0,
        // Buffers for non-streaming clients
        bufferedContent: "",
        bufferedToolCalls: [], // [{id, type:"function", function:{name, arguments}}]
        finishReason: null,
        failed: null,
    };
}

function responsesChunk(state, delta, finishReason) {
    return {
        id: state.chatId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: state.model,
        choices: [{ index: 0, delta: delta || {}, finish_reason: finishReason ?? null }],
    };
}

// Centralized events that should close a <think> block before handling the next event
const RESPONSES_THINKING_STOP_EVENTS = new Set([
    "response.output_text.delta",
    "response.output_item.added",
]);

// Convert a single Azure Responses SSE event into zero or more OpenAI chunks.
// Mirrors ResponseAdapter in Cursor-Azure-GPT-5/app/azure/response_adapter.py.
function translateResponsesEvent(eventName, obj, state) {
    const out = [];

    // Close any open <think> block when a non-reasoning event starts
    if (state.thinking && RESPONSES_THINKING_STOP_EVENTS.has(eventName)) {
        if (GPT_CONFIG.EMIT_THINK_TAGS) {
            out.push(responsesChunk(state, { role: "assistant", content: "</think>\n\n" }));
            state.bufferedContent += "</think>\n\n";
        }
        state.thinking = false;
    }

    switch (eventName) {
        case "response.output_item.added": {
            const itemType = obj?.item?.type;
            if (itemType === "reasoning") {
                state.thinking = true;
                if (GPT_CONFIG.EMIT_THINK_TAGS) {
                    out.push(responsesChunk(state, { role: "assistant", content: "<think>\n\n" }));
                    state.bufferedContent += "<think>\n\n";
                }
            } else if (itemType === "function_call") {
                state.toolCalls += 1;
                const name = obj.item.name || "";
                const args = obj.item.arguments || "";
                const callId = obj.item.call_id || "";
                out.push(responsesChunk(state, {
                    role: "assistant",
                    content: null,
                    tool_calls: [{
                        index: state.toolCalls - 1,
                        id: callId,
                        type: "function",
                        function: { name, arguments: args },
                    }],
                }));
                state.bufferedToolCalls.push({
                    id: callId,
                    type: "function",
                    function: { name, arguments: args },
                });
            }
            break;
        }
        case "response.function_call_arguments.delta": {
            const d = typeof obj?.delta === "string" ? obj.delta : "";
            out.push(responsesChunk(state, {
                tool_calls: [{ index: state.toolCalls - 1, function: { arguments: d } }],
            }));
            const last = state.bufferedToolCalls[state.bufferedToolCalls.length - 1];
            if (last) last.function.arguments = (last.function.arguments || "") + d;
            break;
        }
        case "response.reasoning_summary_text.delta": {
            const d = typeof obj?.delta === "string" ? obj.delta : "";
            out.push(responsesChunk(state, { role: "assistant", content: d }));
            state.bufferedContent += d;
            break;
        }
        case "response.reasoning_summary_text.done": {
            out.push(responsesChunk(state, { role: "assistant", content: "\n\n" }));
            state.bufferedContent += "\n\n";
            break;
        }
        case "response.output_text.delta": {
            const d = typeof obj?.delta === "string" ? obj.delta : "";
            out.push(responsesChunk(state, { role: "assistant", content: d }));
            state.bufferedContent += d;
            break;
        }
        case "response.failed": {
            const err = obj?.response?.error || {};
            const msg = `Azure Responses API error [${err.code || "unknown"}]: ${err.message || "no message"}`;
            state.failed = msg;
            out.push(responsesChunk(state, { role: "assistant", content: msg }));
            state.bufferedContent += msg;
            break;
        }
        default:
            // ignore: response.created, response.in_progress, response.content_part.*,
            // response.output_item.done, response.output_text.done, response.completed, …
            break;
    }
    return out;
}

async function handleGptResponsesApi(req, res) {
    const requestStart = Date.now();
    const cursorModel = req.body?.model;
    const effort = extractGptReasoningEffort(cursorModel);
    const clientWantsStream = req.body?.stream === true;
    const abortController = new AbortController();
    let preStreamHeartbeat = null;

    if (clientWantsStream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();
        res.write(": stream connected\n\n");
        preStreamHeartbeat = setInterval(() => {
            if (!res.writableEnded) {
                try { res.write(": heartbeat\n\n"); } catch {}
            }
        }, 3000);
    }
    const clearHb = () => {
        if (preStreamHeartbeat) { clearInterval(preStreamHeartbeat); preStreamHeartbeat = null; }
    };

    res.on("close", () => {
        clearHb();
        if (!res.writableFinished) {
            console.log(`[GPT-RESP] Client disconnected after ${Date.now() - requestStart}ms, aborting upstream`);
            abortController.abort();
        }
    });

    try {
        const body = buildResponsesBody(req.body, effort);
        const url = buildGptResponsesUrl(GPT_CONFIG.SHAPE, GPT_CONFIG.API_VERSION);

        // Raw-body summary so we can see what Cursor sent (roles, content shapes)
        // without dumping sensitive prompt text. Activated whenever LOG_MESSAGES=true
        // OR when the request arrived with no convertible input.
        const rawMsgs = Array.isArray(req.body?.messages) ? req.body.messages : [];
        const inputSummary = body.input.map(it => it.type || it.role || "?").join(",");
        if (LOG_MESSAGES || body.input.length === 0 || rawMsgs.length === 0) {
            const roleCounts = rawMsgs.reduce((acc, m) => {
                const k = m && m.role ? m.role : "missing";
                acc[k] = (acc[k] || 0) + 1;
                return acc;
            }, {});
            console.log(`[GPT-RESP RAW] messages=${rawMsgs.length}, roles=${JSON.stringify(roleCounts)}, has_tools=${Array.isArray(req.body?.tools) && req.body.tools.length > 0}, has_tool_choice=${!!req.body?.tool_choice}, has_instructions=${!!body.instructions}`);
        }
        console.log(`[GPT-RESP] cursor_model=${cursorModel} → deployment=${GPT_CONFIG.DEPLOYMENT}, effort=${effort}, client_stream=${clientWantsStream}, tools=${body.tools?.length || 0}, input_items=${body.input.length} [${inputSummary}], summary=${body.reasoning.summary || "off"}, verbosity=${body.text?.verbosity || "default"}`);

        const response = await axios.post(url, body, {
            headers: {
                "Content-Type": "application/json",
                "api-key": GPT_CONFIG.API_KEY,
            },
            timeout: 300000,
            responseType: "stream",
            validateStatus: (s) => s < 600,
            signal: abortController.signal,
        });

        console.log(`[GPT-RESP] Azure responded in ${Date.now() - requestStart}ms, status=${response.status}`);
        clearHb();

        if (response.status >= 400) {
            const errorBody = await extractErrorFromResponse(response, true);
            console.error(`[GPT-RESP ERROR] Azure ${response.status}:`, errorBody);
            console.error(`[GPT-RESP ERROR] Request body we sent:`, JSON.stringify({
                model: body.model,
                has_input: Array.isArray(body.input) && body.input.length > 0,
                input_shape: inputSummary,
                has_instructions: !!body.instructions,
                tools_count: body.tools?.length || 0,
                reasoning: body.reasoning,
                has_text: !!body.text,
                truncation: body.truncation,
                stream: body.stream,
            }));
            if (clientWantsStream) return writeSSEError(res, errorBody.error?.message || errorBody.message || "Azure Responses API error");
            return res.status(response.status).json({
                error: {
                    message: errorBody.error?.message || errorBody.message || "Azure Responses API error",
                    type: errorBody.error?.type || "api_error",
                    code: response.status,
                },
            });
        }

        const state = createResponsesStreamState(cursorModel || GPT_CONFIG.DEPLOYMENT);
        let sseBuffer = "";
        let currentEvent = null;
        let chunksForwarded = 0;
        let eventsSeen = 0;

        const writeIfStreaming = (chunk) => {
            if (!clientWantsStream) return;
            if (res.writableEnded) return;
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        };

        const handleSsePayload = (eventName, payload) => {
            eventsSeen += 1;
            let obj = null;
            try { obj = JSON.parse(payload); } catch { return; }
            if (!eventName) eventName = obj?.type || null;
            if (!eventName) return;
            const chunks = translateResponsesEvent(eventName, obj, state);
            for (const c of chunks) {
                chunksForwarded += 1;
                writeIfStreaming(c);
            }
        };

        const processLines = (lines) => {
            for (const rawLine of lines) {
                const line = rawLine.replace(/\r$/, "");
                if (line === "") {
                    currentEvent = null;
                    continue;
                }
                if (line.startsWith(":")) continue;
                if (line.startsWith("event:")) {
                    currentEvent = line.slice(6).trim();
                    continue;
                }
                if (!line.startsWith("data:")) continue;
                const payload = line.slice(5).trim();
                if (!payload || payload === "[DONE]") continue;
                handleSsePayload(currentEvent, payload);
            }
        };

        response.data.on("data", (chunk) => {
            sseBuffer += chunk.toString("utf8");
            const lines = sseBuffer.split("\n");
            sseBuffer = lines.pop() ?? "";
            processLines(lines);
        });

        response.data.on("error", (err) => {
            console.error("[GPT-RESP STREAM] Error:", err.message);
            if (clientWantsStream && !res.writableEnded) {
                try { writeSSEError(res, "Stream error: " + err.message); } catch {}
            } else if (!res.headersSent) {
                try { res.status(502).json({ error: { message: err.message, type: "upstream_error" } }); } catch {}
            }
        });

        response.data.on("end", () => {
            if (sseBuffer.trim().length > 0) {
                processLines(sseBuffer.split("\n"));
                sseBuffer = "";
            }
            // Close any dangling <think>
            if (state.thinking && GPT_CONFIG.EMIT_THINK_TAGS) {
                const closing = responsesChunk(state, { role: "assistant", content: "</think>\n\n" });
                writeIfStreaming(closing);
                state.bufferedContent += "</think>\n\n";
                state.thinking = false;
            }
            const finishReason = state.toolCalls > 0 ? "tool_calls" : "stop";
            state.finishReason = finishReason;

            if (clientWantsStream) {
                writeIfStreaming(responsesChunk(state, {}, finishReason));
                if (!res.writableEnded) {
                    try { res.write("data: [DONE]\n\n"); } catch {}
                    try { res.end(); } catch {}
                }
                console.log(`[GPT-RESP STREAM] Done: events=${eventsSeen}, chunks=${chunksForwarded}, tool_calls=${state.toolCalls}, finish=${finishReason}`);
            } else {
                // Build a single non-streaming chat.completion response for Cursor
                const message = { role: "assistant", content: state.bufferedContent || null };
                if (state.bufferedToolCalls.length > 0) message.tool_calls = state.bufferedToolCalls;
                const payload = {
                    id: state.chatId,
                    object: "chat.completion",
                    created: Math.floor(Date.now() / 1000),
                    model: state.model,
                    choices: [{ index: 0, message, finish_reason: finishReason }],
                    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
                };
                try { res.json(payload); } catch {}
                console.log(`[GPT-RESP JSON] Done: events=${eventsSeen}, tool_calls=${state.toolCalls}, finish=${finishReason}, content_len=${state.bufferedContent.length}`);
            }
        });

        res.on("close", () => {
            if (typeof response.data.destroy === "function") {
                try { response.data.destroy(); } catch {}
            }
        });
    } catch (error) {
        clearHb();
        if (error.code === "ERR_CANCELED" || error.name === "CanceledError") {
            console.log(`[GPT-RESP] Request aborted after ${Date.now() - requestStart}ms (client disconnected)`);
            return;
        }
        console.error(`[GPT-RESP ERROR] After ${Date.now() - requestStart}ms:`, error.message);
        if (clientWantsStream && !res.writableEnded) return writeSSEError(res, error.message);
        if (res.headersSent) return;
        if (error.response) {
            return res.status(error.response.status).json({
                error: { message: error.response.data?.error?.message || error.message, type: "api_error", code: error.response.status },
            });
        }
        if (error.request) {
            return res.status(503).json({ error: { message: "Unable to reach Azure Responses API: " + error.message, type: "connection_error" } });
        }
        return res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
    }
}

// ─── Azure OpenAI (GPT-5.4) Passthrough ─────────────────────────────────────
//
// gpt-5.4 supports the Chat Completions API natively (per Azure Foundry docs,
// March 2026), so we can near-passthrough Cursor's OpenAI-format request. Only
// tweaks: swap `model` → Azure deployment, convert max_tokens →
// max_completion_tokens (required for reasoning models), strip temperature/top_p
// (unsupported on reasoning models), and inject `reasoning_effort` from the
// Cursor model-name suffix (e.g. gpt-5.4-high → "high").
async function handleGptChatCompletions(req, res) {
    const requestStart = Date.now();

    if (!GPT_CONFIG.SHAPE) {
        return res.status(500).json({
            error: { message: "AZURE_OPENAI_ENDPOINT not configured", type: "configuration_error" },
        });
    }
    if (!GPT_CONFIG.API_KEY) {
        return res.status(500).json({
            error: { message: "AZURE_OPENAI_API_KEY / AZURE_API_KEY not configured", type: "configuration_error" },
        });
    }

    const cursorModel = req.body?.model;

    // Fast-path validation pings (same logic as Anthropic route)
    if (isModelValidationPing(req.body)) {
        console.log(`[GPT] Model validation ping (model=${cursorModel}), responding locally`);
        return res.json(makeValidationResponse(cursorModel || GPT_CONFIG.DEPLOYMENT));
    }

    const effort = extractGptReasoningEffort(cursorModel);
    const isStreaming = req.body?.stream === true;
    const abortController = new AbortController();
    let preStreamHeartbeat = null;

    if (isStreaming) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();
        res.write(": stream connected\n\n");
        preStreamHeartbeat = setInterval(() => {
            if (!res.writableEnded) {
                try { res.write(": heartbeat\n\n"); } catch {}
            }
        }, 3000);
    }

    function clearPreStreamHeartbeat() {
        if (preStreamHeartbeat) {
            clearInterval(preStreamHeartbeat);
            preStreamHeartbeat = null;
        }
    }

    res.on("close", () => {
        clearPreStreamHeartbeat();
        if (!res.writableFinished) {
            console.log(`[GPT] Client disconnected after ${Date.now() - requestStart}ms, aborting upstream`);
            abortController.abort();
        }
    });

    try {
        const upstreamBody = { ...req.body };
        upstreamBody.model = GPT_CONFIG.DEPLOYMENT;

        // Reasoning models require max_completion_tokens (max_tokens is deprecated for them)
        if (upstreamBody.max_tokens != null && upstreamBody.max_completion_tokens == null) {
            upstreamBody.max_completion_tokens = upstreamBody.max_tokens;
        }
        delete upstreamBody.max_tokens;

        // Unsupported sampling params on gpt-5.x reasoning models
        delete upstreamBody.temperature;
        delete upstreamBody.top_p;
        delete upstreamBody.presence_penalty;
        delete upstreamBody.frequency_penalty;

        if (!upstreamBody.reasoning_effort) {
            upstreamBody.reasoning_effort = effort;
        }

        const url = buildGptUrl(GPT_CONFIG.SHAPE, GPT_CONFIG.DEPLOYMENT, GPT_CONFIG.API_VERSION);

        console.log(`[GPT] cursor_model=${cursorModel} → deployment=${GPT_CONFIG.DEPLOYMENT}, effort=${upstreamBody.reasoning_effort}, stream=${isStreaming}, tools=${upstreamBody.tools?.length || 0}, messages=${upstreamBody.messages?.length || 0}, api_style=${GPT_CONFIG.SHAPE.style}`);

        const response = await axios.post(url, upstreamBody, {
            headers: {
                "Content-Type": "application/json",
                "api-key": GPT_CONFIG.API_KEY,
            },
            timeout: 300000,
            responseType: isStreaming ? "stream" : "json",
            validateStatus: (s) => s < 600,
            signal: abortController.signal,
        });

        const azureElapsed = Date.now() - requestStart;
        console.log(`[GPT] Azure responded in ${azureElapsed}ms, status=${response.status}`);
        clearPreStreamHeartbeat();

        if (response.status >= 400) {
            const errorBody = await extractErrorFromResponse(response, isStreaming);
            console.error(`[GPT ERROR] Azure ${response.status}:`, errorBody);
            if (isStreaming) {
                return writeSSEError(res, errorBody.error?.message || errorBody.message || "Azure OpenAI error");
            }
            return res.status(response.status).json({
                error: {
                    message: errorBody.error?.message || errorBody.message || "Azure OpenAI error",
                    type: errorBody.error?.type || "api_error",
                    code: response.status,
                },
            });
        }

        if (isStreaming) {
            // Azure OpenAI SSE is *almost* OpenAI-compatible, but Azure:
            //   1) injects a non-standard prologue chunk (empty `choices` +
            //      `prompt_filter_results`) that some clients reject as malformed;
            //   2) rewrites `model` to the dated deployment name
            //      (e.g. `gpt-5.4-2026-03-05`) instead of the id Cursor asked for,
            //      which strict clients may reject as a model-mismatch;
            //   3) sprays extra fields (`content_filter_results`, `obfuscation`,
            //      `service_tier`, `system_fingerprint`, `prompt_filter_results`).
            // This sanitizer fixes all three line-by-line before forwarding.
            const originalModelId = cursorModel || GPT_CONFIG.DEPLOYMENT;
            let sseBuffer = "";
            let prologueDropped = false;
            let chunksForwarded = 0;
            let modelRewrites = 0;

            const handleSseChunk = (chunk) => {
                sseBuffer += chunk.toString("utf8");
                // SSE events are separated by a blank line. We split on \n to handle
                // partial frames; we keep the last (possibly incomplete) line in the buffer.
                const lines = sseBuffer.split("\n");
                sseBuffer = lines.pop() ?? "";
                for (const rawLine of lines) {
                    const line = rawLine.replace(/\r$/, "");

                    if (line === "") {
                        // frame separator — just pass through
                        if (!res.writableEnded) res.write("\n");
                        continue;
                    }
                    if (line.startsWith(":")) {
                        // SSE comment / heartbeat — pass through
                        if (!res.writableEnded) res.write(`${line}\n`);
                        continue;
                    }
                    if (!line.startsWith("data:")) {
                        // unknown SSE field (event:, id:, retry:) — pass through
                        if (!res.writableEnded) res.write(`${line}\n`);
                        continue;
                    }

                    const payload = line.slice(5).trim();
                    if (payload === "[DONE]") {
                        if (!res.writableEnded) res.write(`data: [DONE]\n`);
                        continue;
                    }

                    let obj;
                    try {
                        obj = JSON.parse(payload);
                    } catch {
                        // Unparseable — forward verbatim to avoid losing data
                        if (!res.writableEnded) res.write(`${line}\n`);
                        continue;
                    }

                    // 1) Drop Azure's prologue: empty choices + filter metadata only.
                    const isPrologue =
                        Array.isArray(obj.choices) &&
                        obj.choices.length === 0 &&
                        (obj.prompt_filter_results !== undefined || obj.id === "" || obj.model === "");
                    if (isPrologue) {
                        prologueDropped = true;
                        continue;
                    }

                    // 2) Strip Azure-only noise fields from the chunk.
                    delete obj.prompt_filter_results;
                    delete obj.obfuscation;
                    delete obj.system_fingerprint;
                    delete obj.service_tier;
                    if (Array.isArray(obj.choices)) {
                        for (const choice of obj.choices) {
                            if (choice && typeof choice === "object") {
                                delete choice.content_filter_results;
                                delete choice.content_filter_offsets;
                                if (choice.delta && typeof choice.delta === "object") {
                                    // `refusal: null` is harmless but not standard; strip to be safe.
                                    if (choice.delta.refusal === null) delete choice.delta.refusal;
                                }
                            }
                        }
                    }

                    // 3) Rewrite `model` back to the id Cursor originally requested.
                    // Azure returns the dated variant (e.g. gpt-5.4-2026-03-05) which
                    // strict OpenAI clients may reject as a mismatch against the
                    // model they asked for.
                    if (typeof obj.model === "string" && obj.model !== originalModelId) {
                        obj.model = originalModelId;
                        modelRewrites += 1;
                    }

                    chunksForwarded += 1;
                    if (!res.writableEnded) {
                        res.write(`data: ${JSON.stringify(obj)}\n`);
                    }
                }
            };

            response.data.on("data", handleSseChunk);
            response.data.on("end", () => {
                // Flush any leftover partial line (rare, should be empty).
                if (sseBuffer.trim().length > 0 && !res.writableEnded) {
                    res.write(sseBuffer);
                    sseBuffer = "";
                }
                if (!res.writableEnded) res.end();
                console.log(`[GPT STREAM] Done: prologue_dropped=${prologueDropped}, chunks_forwarded=${chunksForwarded}, model_rewrites=${modelRewrites}`);
            });
            response.data.on("error", (err) => {
                console.error("[GPT STREAM] Error:", err.message);
                if (!res.writableEnded) {
                    try { res.end(); } catch {}
                }
            });
            res.on("close", () => {
                if (typeof response.data.destroy === "function") {
                    try { response.data.destroy(); } catch {}
                }
            });
        } else {
            const originalModelId = cursorModel || GPT_CONFIG.DEPLOYMENT;
            const sanitized = { ...response.data };
            delete sanitized.prompt_filter_results;
            delete sanitized.system_fingerprint;
            delete sanitized.service_tier;
            if (typeof sanitized.model === "string" && sanitized.model !== originalModelId) {
                sanitized.model = originalModelId;
            }
            if (Array.isArray(sanitized.choices)) {
                sanitized.choices = sanitized.choices.map((c) => {
                    if (!c || typeof c !== "object") return c;
                    const copy = { ...c };
                    delete copy.content_filter_results;
                    delete copy.content_filter_offsets;
                    if (copy.message && typeof copy.message === "object" && copy.message.refusal === null) {
                        copy.message = { ...copy.message };
                        delete copy.message.refusal;
                    }
                    return copy;
                });
            }
            res.json(sanitized);
        }
    } catch (error) {
        clearPreStreamHeartbeat();
        if (error.code === "ERR_CANCELED" || error.name === "CanceledError") {
            console.log(`[GPT] Request aborted after ${Date.now() - requestStart}ms (client disconnected)`);
            return;
        }
        console.error(`[GPT ERROR] After ${Date.now() - requestStart}ms:`, error.message);
        if (isStreaming && !res.writableEnded) {
            return writeSSEError(res, error.message);
        }
        if (res.headersSent) return;
        if (error.response) {
            return res.status(error.response.status).json({
                error: { message: error.response.data?.error?.message || error.message, type: "api_error", code: error.response.status },
            });
        }
        if (error.request) {
            return res.status(503).json({ error: { message: "Unable to reach Azure OpenAI: " + error.message, type: "connection_error" } });
        }
        return res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
    }
}

async function handleChatCompletions(req, res) {
    // Route GPT-family requests to Azure OpenAI; keep Claude on the Anthropic endpoint.
    if (isGptModel(req.body?.model)) {
        // Short-circuit model-validation pings locally regardless of API mode,
        // so Cursor's "Verify" flow works even if Azure Responses is slow or
        // upstream creds are still being configured.
        if (isModelValidationPing(req.body)) {
            console.log(`[GPT] Model validation ping (model=${req.body.model}), responding locally`);
            return res.json(makeValidationResponse(req.body.model || GPT_CONFIG.DEPLOYMENT));
        }
        if (GPT_CONFIG.API_MODE === "responses") {
            if (!GPT_CONFIG.SHAPE) {
                return res.status(500).json({ error: { message: "AZURE_OPENAI_ENDPOINT not configured", type: "configuration_error" } });
            }
            if (!GPT_CONFIG.API_KEY) {
                return res.status(500).json({ error: { message: "AZURE_OPENAI_API_KEY / AZURE_API_KEY not configured", type: "configuration_error" } });
            }
            return handleGptResponsesApi(req, res);
        }
        return handleGptChatCompletions(req, res);
    }

    const requestStart = Date.now();

    // Fast-path: respond instantly to Cursor's model-validation pings
    // (non-streaming, no tools, 1-2 messages) to prevent disconnect during Azure cold start
    if (isModelValidationPing(req.body)) {
        const validationModel = resolveDeployment(req.body.model);
        console.log(`[PROXY] Model validation ping (model=${req.body.model} → ${validationModel}), responding locally`);
        return res.json(makeValidationResponse(validationModel));
    }

    const abortController = new AbortController();
    const isStreaming = req.body?.stream === true;
    let preStreamHeartbeat = null;

    // For streaming: establish SSE connection immediately so the client
    // knows we're alive while waiting for Azure (which can be slow for thinking models)
    if (isStreaming) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();
        // Write immediate body data so Railway's proxy forwards to the client right away
        res.write(": stream connected\n\n");

        preStreamHeartbeat = setInterval(() => {
            if (!res.writableEnded) {
                try { res.write(": heartbeat\n\n"); } catch {}
            }
        }, 3000);
    }

    function clearPreStreamHeartbeat() {
        if (preStreamHeartbeat) {
            clearInterval(preStreamHeartbeat);
            preStreamHeartbeat = null;
        }
    }

    // Use res.on("close") instead of req.on("close") — req's close event fires
    // after express.json() consumes the body (Node.js 18+), causing false disconnects.
    // res.on("close") fires when the actual TCP connection drops.
    res.on("close", () => {
        clearPreStreamHeartbeat();
        if (!res.writableFinished) {
            const elapsed = Date.now() - requestStart;
            console.log(`[PROXY] Client disconnected after ${elapsed}ms, aborting upstream request`);
            abortController.abort();
        }
    });

    try {
        if (!CONFIG.AZURE_API_KEY) {
            clearPreStreamHeartbeat();
            if (isStreaming) return writeSSEError(res, "Azure API key not configured", "configuration_error");
            return res.status(500).json({ error: { message: "Azure API key not configured", type: "configuration_error" } });
        }
        if (!CONFIG.AZURE_ENDPOINT) {
            clearPreStreamHeartbeat();
            if (isStreaming) return writeSSEError(res, "Azure endpoint not configured", "configuration_error");
            return res.status(500).json({ error: { message: "Azure endpoint not configured", type: "configuration_error" } });
        }
        if (!req.body || (!req.body.messages && !req.body.input)) {
            clearPreStreamHeartbeat();
            if (isStreaming) return writeSSEError(res, "Invalid request: missing messages", "invalid_request_error");
            return res.status(400).json({ error: { message: "Invalid request: missing messages", type: "invalid_request_error" } });
        }

        // Compact diagnostic: count tool-related fields in raw messages from Cursor
        if (LOG_MESSAGES) {
            const rawMsgs = req.body.messages || [];
            let nToolCalls = 0, nToolRole = 0, nFunctionRole = 0, nFunctionCall = 0, nContentToolUse = 0, nContentToolResult = 0;
            for (const m of rawMsgs) {
                if (m.tool_calls && m.tool_calls.length > 0) nToolCalls++;
                if (m.role === "tool") nToolRole++;
                if (m.role === "function") nFunctionRole++;
                if (m.function_call) nFunctionCall++;
                if (Array.isArray(m.content)) {
                    for (const p of m.content) {
                        if (p.type === "tool_use") nContentToolUse++;
                        if (p.type === "tool_result") nContentToolResult++;
                    }
                }
            }
            console.log(`[RAW_SUMMARY] total=${rawMsgs.length}, tool_calls_field=${nToolCalls}, role_tool=${nToolRole}, role_function=${nFunctionRole}, function_call_field=${nFunctionCall}, content_tool_use=${nContentToolUse}, content_tool_result=${nContentToolResult}`);

            // Log details of the LAST 6 messages (most likely to contain the current tool interaction)
            const startIdx = Math.max(0, rawMsgs.length - 6);
            for (let i = startIdx; i < rawMsgs.length; i++) {
                const m = rawMsgs[i];
                const parts = [`role=${m.role}`];
                if (m.content === null || m.content === undefined) {
                    parts.push("content=null");
                } else if (typeof m.content === "string") {
                    parts.push(`content=string(${m.content.length}ch)`);
                } else if (Array.isArray(m.content)) {
                    const blockTypes = m.content.map(p => p.type || "?").join(",");
                    parts.push(`content=[${blockTypes}](${m.content.length} blocks)`);
                }
                if (m.tool_calls) parts.push(`tool_calls=${JSON.stringify(m.tool_calls.map(tc => ({ id: tc.id?.substring(0, 15), fn: tc.function?.name })))}`);
                if (m.tool_call_id) parts.push(`tool_call_id=${m.tool_call_id.substring(0, 20)}`);
                if (m.function_call) parts.push(`function_call=${m.function_call.name}`);
                if (m.name) parts.push(`name=${m.name}`);
                console.log(`[RAW_TAIL] [${i}] ${parts.join(", ")}`);
            }
        }

        const anthropicRequest = buildAnthropicRequest(req.body);
        // Use the resolved deployment name (e.g. "claude-opus-4-6") in responses
        // back to Cursor, NOT the raw alias (e.g. "opus46"). Cursor stores the
        // model id from responses and uses it for internal operations like resuming
        // chats after background tasks. Aliases like "opus46" are not recognized
        // by Cursor's own model resolver and cause "AI model not found" errors.
        const responseModelId = anthropicRequest.model;

        console.log(`[PROXY] ── Request ──────────────────────────────────`);
        console.log(`[PROXY] cursor_model=${req.body.model} → deployment=${anthropicRequest.model}`);
        console.log(`[PROXY] cursor_max_tokens=${req.body.max_tokens || req.body.max_completion_tokens || 'not set'} → actual_max_tokens=${anthropicRequest.max_tokens}`);
        console.log(`[PROXY] stream=${isStreaming}, tools=${anthropicRequest.tools?.length || 0}, messages=${anthropicRequest.messages.length}, tool_choice=${JSON.stringify(anthropicRequest.tool_choice || 'none')}${anthropicRequest.thinking ? ', thinking=adaptive(effort=' + (anthropicRequest.output_config?.effort || 'high') + ')' : ''}`);

        if (LOG_MESSAGES) {
            for (let i = 0; i < anthropicRequest.messages.length; i++) {
                const m = anthropicRequest.messages[i];
                const contentSummary = Array.isArray(m.content)
                    ? m.content.map(b => {
                        if (b.type === "text") return `text(${b.text?.length || 0}ch)`;
                        if (b.type === "thinking") return `thinking(${b.thinking?.length || 0}ch)`;
                        if (b.type === "redacted_thinking") return `redacted_thinking`;
                        if (b.type === "tool_use") return `tool_use(${b.name})`;
                        if (b.type === "tool_result") return `tool_result(id=${b.tool_use_id?.substring(0, 12)}…)`;
                        return b.type || "unknown";
                    }).join(", ")
                    : typeof m.content === "string" ? `string(${m.content.length}ch)` : String(m.content);
                console.log(`[MSG] [${i}] role=${m.role}, content=[${contentSummary}]`);
            }
        }

        console.log(`[PROXY] Calling Azure endpoint...`);

        // Retry logic for transient errors (429 rate limit, 529 overloaded)
        const MAX_RETRIES = 3;
        let response;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            const reqHeaders = {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": CONFIG.ANTHROPIC_VERSION,
            };
            // Build beta flags from Azure-supported list + conditional flags
            const betaFlags = [...AZURE_SUPPORTED_BETA_FLAGS];
            if (anthropicRequest.thinking && anthropicRequest.tools) {
                betaFlags.push("interleaved-thinking-2025-05-14");
            }
            if (betaFlags.length > 0) {
                reqHeaders["anthropic-beta"] = betaFlags.join(",");
            }

            response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
                headers: reqHeaders,
                timeout: 300000,
                responseType: isStreaming ? "stream" : "json",
                validateStatus: (status) => status < 600,
                signal: abortController.signal,
            });

            // Retry on 429 (rate limited) or 529 (overloaded) — common during parallel tool calls
            if ((response.status === 429 || response.status === 529) && attempt < MAX_RETRIES) {
                const retryAfter = parseInt(response.headers["retry-after"]) || (attempt * 2);
                console.log(`[PROXY] Got ${response.status}, retry ${attempt}/${MAX_RETRIES} after ${retryAfter}s`);
                // Drain error stream body to avoid memory leak
                if (isStreaming && typeof response.data?.destroy === "function") {
                    response.data.destroy();
                }
                await new Promise(r => setTimeout(r, retryAfter * 1000));
                continue;
            }
            break;
        }

        const azureElapsed = Date.now() - requestStart;
        console.log(`[PROXY] Azure responded in ${azureElapsed}ms, status=${response.status}`);

        clearPreStreamHeartbeat();

        if (response.status >= 400) {
            const errorBody = await extractErrorFromResponse(response, isStreaming);
            console.error(`[ERROR] Azure ${response.status}:`, errorBody);
            if (isStreaming) return writeSSEError(res, errorBody.error?.message || errorBody.message || "Azure API error");
            return res.status(response.status).json({
                error: {
                    message: errorBody.error?.message || errorBody.message || "Azure API error",
                    type: errorBody.error?.type || "api_error",
                    code: response.status,
                },
            });
        }

        if (isStreaming) {
            handleAnthropicStream(response, res, responseModelId, abortController, true);
        } else {
            const anthropicStopReason = response.data?.stop_reason;
            const openaiResponse = convertAnthropicResponseToOpenai(response.data, responseModelId);
            console.log(`[RESPONSE] anthropic_stop=${anthropicStopReason} → finish_reason=${openaiResponse.choices[0].finish_reason}, tool_calls=${openaiResponse.choices[0].message.tool_calls?.length || 0}, usage=${JSON.stringify(openaiResponse.usage)}`);
            if (anthropicStopReason === "max_tokens") {
                console.log(`[RESPONSE] ⚠️  OUTPUT TRUNCATED — model hit max_tokens. cursor_max_tokens=${req.body.max_tokens || 'not set'}, actual=${anthropicRequest.max_tokens}`);
            }
            res.json(openaiResponse);
        }
    } catch (error) {
        clearPreStreamHeartbeat();
        if (error.code === "ERR_CANCELED" || error.name === "CanceledError") {
            console.log(`[PROXY] Request aborted after ${Date.now() - requestStart}ms (client disconnected)`);
            return;
        }
        console.error(`[ERROR] After ${Date.now() - requestStart}ms:`, error.message);
        if (isStreaming && !res.writableEnded) {
            return writeSSEError(res, error.message);
        }
        if (res.headersSent) return;
        if (error.response) {
            return res.status(error.response.status).json({
                error: { message: error.response.data?.error?.message || error.message, type: "api_error", code: error.response.status },
            });
        }
        if (error.request) {
            return res.status(503).json({ error: { message: "Unable to reach Azure API: " + error.message, type: "connection_error" } });
        }
        return res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
    }
}

async function extractErrorFromResponse(response, isStreaming) {
    if (isStreaming && typeof response.data?.pipe === "function") {
        return new Promise((resolve) => {
            let buf = "";
            response.data.on("data", (chunk) => { buf += chunk.toString(); });
            response.data.on("end", () => {
                try { resolve(JSON.parse(buf)); } catch { resolve({ message: buf }); }
            });
            response.data.on("error", () => resolve({ message: "Error reading error stream" }));
        });
    }
    if (typeof response.data === "string") {
        try { return JSON.parse(response.data); } catch { return { message: response.data }; }
    }
    return response.data || { message: "Unknown error" };
}

// ─── Routes ──────────────────────────────────────────────────────────────────

app.get("/", (req, res) => {
    res.json({
        status: "running",
        name: "Cursor Azure Anthropic Proxy",
        endpoints: ["/v1/chat/completions", "/v1/models", "/health"],
    });
});

app.get("/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString(), apiKeyConfigured: !!CONFIG.AZURE_API_KEY });
});

app.post("/chat/completions", requireAuth, handleChatCompletions);
app.post("/v1/chat/completions", requireAuth, handleChatCompletions);

function getModelList() {
    const models = [];
    const seen = new Set();
    // Advertise resolved deployment names, NOT raw aliases (e.g. "opus46").
    // Cursor stores the model id from responses and uses it for internal
    // operations; unrecognized aliases cause "AI model not found" on resume.
    const claudeModels = [
        ...Object.values(MODEL_MAP),
        ...CLAUDE_DIRECT_DEPLOYMENTS,
        ...Object.values(CLAUDE_ALIAS_MAP),
        "claude-opus-4-6",
        "claude-opus-4-7",
    ];
    for (const deployment of claudeModels) {
        if (!seen.has(deployment)) {
            seen.add(deployment);
            // Cursor consumes this through its OpenAI-compatible custom-model path,
            // so advertise Claude deployments as OpenAI-style model ids.
            models.push({ id: deployment, object: "model", created: 1700000000, owned_by: "openai" });
        }
    }
    // Advertise the gpt-5.4 reasoning-effort variants when an Azure OpenAI
    // endpoint is configured. The base id can still be used without a suffix.
    if (GPT_CONFIG.SHAPE) {
        for (const id of Object.keys(GPT_MODEL_MAP)) {
            if (!seen.has(id)) {
                seen.add(id);
                models.push({ id, object: "model", created: 1700000000, owned_by: "openai" });
            }
        }
    }
    return { object: "list", data: models };
}

app.get("/v1/models", (req, res) => res.json(getModelList()));
app.get("/models", (req, res) => res.json(getModelList()));

// Anthropic-native passthrough for /v1/messages
app.post("/v1/messages", requireAuth, async (req, res) => {
    const abortController = new AbortController();
    const isStreaming = req.body?.stream === true;
    let preStreamHeartbeat = null;

    if (isStreaming) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();
        res.write(": stream connected\n\n");

        preStreamHeartbeat = setInterval(() => {
            if (!res.writableEnded) {
                try { res.write(": heartbeat\n\n"); } catch {}
            }
        }, 3000);
    }

    function clearHeartbeat() {
        if (preStreamHeartbeat) {
            clearInterval(preStreamHeartbeat);
            preStreamHeartbeat = null;
        }
    }

    res.on("close", () => {
        clearHeartbeat();
        if (!res.writableFinished) {
            console.log("[/v1/messages] Client disconnected, aborting upstream");
            abortController.abort();
        }
    });

    try {
        const passthroughHeaders = {
            "Content-Type": "application/json",
            "x-api-key": CONFIG.AZURE_API_KEY,
            "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
        };
        if (req.headers["anthropic-beta"]) {
            passthroughHeaders["anthropic-beta"] = req.headers["anthropic-beta"];
        }
        const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
            headers: passthroughHeaders,
            timeout: 300000,
            responseType: isStreaming ? "stream" : "json",
            signal: abortController.signal,
        });

        clearHeartbeat();

        if (isStreaming) {
            response.data.on("error", (error) => {
                console.error("[/v1/messages STREAM] Error:", error.message);
                if (!res.writableEnded) {
                    try { res.end(); } catch {}
                }
            });

            res.on("close", () => {
                if (response.data && typeof response.data.destroy === "function") {
                    response.data.destroy();
                }
            });

            response.data.pipe(res);
        } else {
            res.json(response.data);
        }
    } catch (error) {
        clearHeartbeat();
        if (error.name === "CanceledError" || error.code === "ERR_CANCELED") {
            console.log("[/v1/messages] Request aborted due to client disconnect");
            return;
        }
        console.error("[ERROR /v1/messages]", error.message);
        if (isStreaming && !res.writableEnded) {
            try {
                res.write(`data: ${JSON.stringify({ type: "error", error: { message: error.message, type: "proxy_error" } })}\n\n`);
                res.end();
            } catch {}
            return;
        }
        if (!res.headersSent) {
            res.status(error.response?.status || 500).json({
                error: { message: error.message, type: "proxy_error" },
            });
        }
    }
});

app.use((req, res) => {
    res.status(404).json({ error: { message: "Not found. Use /v1/chat/completions or /v1/models", type: "not_found" } });
});

// ─── Server ──────────────────────────────────────────────────────────────────

const server = app.listen(CONFIG.PORT, "0.0.0.0", () => {
    console.log("=".repeat(60));
    console.log(`Cursor Azure Anthropic Proxy`);
    console.log(`Port: ${CONFIG.PORT}`);
    console.log(`Claude Default Deployment: ${DEFAULT_DEPLOYMENT} (AZURE_CLAUDE_DEPLOYMENT_NAME)`);
    console.log(`Claude Model Map: ${JSON.stringify(MODEL_MAP)}`);
    const endpointSource = process.env.AZURE_ENDPOINT === CONFIG.AZURE_ENDPOINT
        ? "AZURE_ENDPOINT"
        : `derived from AZURE_ENDPOINT (${process.env.AZURE_ENDPOINT || "unset"})`;
    console.log(`Claude Endpoint: ${CONFIG.AZURE_ENDPOINT} [${endpointSource}]`);
    console.log(`Thinking: adaptive (effort=${THINKING_EFFORT}, env THINKING_EFFORT)`);
    console.log(`Thinking With Tools: always on (adaptive thinking is tool-aware)`);
    console.log(`Min Output Tokens: ${MIN_OUTPUT_TOKENS}`);
    console.log(`API Key: ${CONFIG.AZURE_API_KEY ? "configured" : "MISSING"}`);
    console.log(`Auth Key: ${CONFIG.SERVICE_API_KEY ? "configured" : "MISSING"}`);
    if (GPT_CONFIG.SHAPE) {
        const chatUrl = buildGptUrl(GPT_CONFIG.SHAPE, GPT_CONFIG.DEPLOYMENT, GPT_CONFIG.API_VERSION);
        const respUrl = buildGptResponsesUrl(GPT_CONFIG.SHAPE, GPT_CONFIG.API_VERSION);
        const activeUrl = GPT_CONFIG.API_MODE === "responses" ? respUrl : chatUrl;
        console.log(`GPT Default Deployment: ${GPT_CONFIG.DEPLOYMENT} (AZURE_GPT_DEPLOYMENT)`);
        console.log(`GPT Model Map (cursor_id → reasoning_effort): ${JSON.stringify(GPT_MODEL_MAP)}`);
        console.log(`GPT API Mode: ${GPT_CONFIG.API_MODE} (AZURE_GPT_API_MODE: chat | responses)`);
        console.log(`GPT Endpoint: ${activeUrl} [mode=${GPT_CONFIG.API_MODE}, default_effort=${GPT_CONFIG.DEFAULT_EFFORT}]`);
        if (GPT_CONFIG.API_MODE === "responses") {
            console.log(`GPT Responses Config: summary=${GPT_CONFIG.SUMMARY_LEVEL}, verbosity=${GPT_CONFIG.VERBOSITY_LEVEL}, truncation=${GPT_CONFIG.TRUNCATION}, emit_think_tags=${GPT_CONFIG.EMIT_THINK_TAGS}`);
        }
    } else {
        console.log(`Azure OpenAI (GPT): disabled (AZURE_OPENAI_ENDPOINT not set)`);
    }
    console.log(`LOG_TOOL_CALLS: ${LOG_TOOL_CALLS}`);
    console.log(`LOG_MESSAGES: ${LOG_MESSAGES}`);
    console.log("=".repeat(60));
});

server.timeout = 300000;
server.keepAliveTimeout = 300000;
server.headersTimeout = 305000;

function gracefulShutdown(signal) {
    console.log(`[${signal}] Graceful shutdown started, waiting for in-flight requests...`);
    const forceExitTimeout = setTimeout(() => {
        console.log(`[${signal}] Force exit after 30s timeout`);
        process.exit(1);
    }, 30000);
    forceExitTimeout.unref();
    server.close(() => {
        console.log(`[${signal}] All connections closed, exiting`);
        process.exit(0);
    });
}

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));
