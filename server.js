const express = require("express");
const axios = require("axios");
const { createStreamState, translateClaudeEvent } = require("./stream-translator");

const app = express();
app.use(express.json({ limit: "250mb" }));

const CONFIG = {
    AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
    AZURE_API_KEY: process.env.AZURE_API_KEY,
    SERVICE_API_KEY: process.env.SERVICE_API_KEY,
    PORT: process.env.PORT || 8080,
    ANTHROPIC_VERSION: "2023-06-01",
};

// ─── Model Routing ───────────────────────────────────────────────────────────

const MODEL_MAP = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-3-5",
};

const DEFAULT_DEPLOYMENT = "claude-sonnet-4-6";

function resolveDeployment(cursorModel) {
    if (!cursorModel) return DEFAULT_DEPLOYMENT;
    const lower = cursorModel.toLowerCase();

    // Direct match in MODEL_MAP families
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

            if (msg.content) {
                if (typeof msg.content === "string") {
                    contentBlocks.push({ type: "text", text: msg.content });
                } else if (Array.isArray(msg.content)) {
                    for (const part of msg.content) {
                        if (part.type === "text") contentBlocks.push({ type: "text", text: part.text });
                    }
                }
            }

            if (msg.tool_calls && msg.tool_calls.length > 0) {
                for (const tc of msg.tool_calls) {
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

            if (contentBlocks.length > 0) {
                anthropicMessages.push({ role: "assistant", content: contentBlocks });
            }
            continue;
        }

        if (msg.role === "tool") {
            const toolResult = {
                type: "tool_result",
                tool_use_id: msg.tool_call_id,
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

function convertToolChoiceToAnthropic(openaiToolChoice) {
    if (!openaiToolChoice) return undefined;
    if (openaiToolChoice === "auto") return { type: "auto" };
    if (openaiToolChoice === "required") return { type: "any" };
    if (openaiToolChoice === "none") return undefined;
    if (typeof openaiToolChoice === "object" && openaiToolChoice.type === "function") {
        return { type: "tool", name: openaiToolChoice.function.name };
    }
    return undefined;
}

function shouldEnableThinking(modelName) {
    if (!modelName) return false;
    const lower = modelName.toLowerCase();
    return lower.includes("thinking") || lower.includes("think");
}

const MODEL_MAX_OUTPUT = {
    "claude-opus-4-6": 32000,
    "claude-sonnet-4-6": 64000,
    "claude-haiku-3-5": 8192,
};
const THINKING_MAX_OUTPUT = 128000;
const MIN_OUTPUT_TOKENS = 16384;
const THINKING_BUDGET_TOKENS = parseInt(process.env.THINKING_BUDGET_TOKENS) || 10000;

function resolveMaxTokens(openaiMaxTokens, deployment, thinkingEnabled) {
    if (thinkingEnabled) return THINKING_MAX_OUTPUT;
    const modelMax = MODEL_MAX_OUTPUT[deployment] || 64000;
    if (!openaiMaxTokens || openaiMaxTokens < MIN_OUTPUT_TOKENS) {
        return Math.min(modelMax, Math.max(MIN_OUTPUT_TOKENS, modelMax));
    }
    return Math.min(openaiMaxTokens, modelMax);
}

function buildAnthropicRequest(openaiBody) {
    const { system, messages } = convertMessagesToAnthropic(openaiBody.messages || []);
    const thinkingEnabled = shouldEnableThinking(openaiBody.model);
    const deployment = resolveDeployment(openaiBody.model);

    const anthropicReq = {
        model: deployment,
        messages,
        max_tokens: resolveMaxTokens(openaiBody.max_tokens, deployment, thinkingEnabled),
    };

    if (system) anthropicReq.system = system;
    if (openaiBody.stream !== undefined) anthropicReq.stream = openaiBody.stream;
    if (openaiBody.stop) anthropicReq.stop_sequences = Array.isArray(openaiBody.stop) ? openaiBody.stop : [openaiBody.stop];

    if (thinkingEnabled) {
        // Keep thinking budget modest — high budgets (50K+) cause the model to
        // overthink and describe actions in text instead of executing tool calls.
        // Configurable via THINKING_BUDGET_TOKENS env var (default: 10000).
        const budgetTokens = THINKING_BUDGET_TOKENS;
        anthropicReq.thinking = { type: "enabled", budget_tokens: budgetTokens };
    } else {
        if (openaiBody.temperature !== undefined) anthropicReq.temperature = openaiBody.temperature;
        if (openaiBody.top_p !== undefined) anthropicReq.top_p = openaiBody.top_p;
    }

    const tools = convertToolsToAnthropic(openaiBody.tools);
    if (tools) anthropicReq.tools = tools;

    const toolChoice = convertToolChoiceToAnthropic(openaiBody.tool_choice);
    if (toolChoice) anthropicReq.tool_choice = toolChoice;

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

    return {
        id: anthropicResp.id || "chatcmpl-" + Date.now(),
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: requestModel || DEFAULT_DEPLOYMENT,
        choices: [{ index: 0, message, finish_reason: anthropicStopToOpenai(anthropicResp.stop_reason) }],
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

const MAX_STREAM_DURATION_MS = 10 * 60 * 1000; // 10 minutes

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

            const chunks = translateClaudeEvent(event, state);
            if (chunks) {
                for (const c of chunks) writeChunk(res, c);
            }

            if (event.type === "message_delta" && state.finishReasonSent) {
                console.log(`[STREAM] Done: finish=${state.finishReason}, tools=${state.toolCalls.size}, usage=${JSON.stringify(state.usage || {})}`);
            }

            if (event.type === "message_stop") {
                streamCompleted = true;
                try { res.write("data: [DONE]\n\n"); } catch {}
                cleanup();
            }
        }
    });

    axiosResponse.data.on("end", () => {
        console.log("[STREAM] Upstream ended");
        if (!streamCompleted && !streamEnded && !res.writableEnded) {
            // Upstream closed without message_stop — send a graceful termination
            if (!state.finishReasonSent) {
                const finishReason = state.toolCalls.size > 0 ? "tool_calls" : "stop";
                writeChunk(res, {
                    id: `chatcmpl-${state.messageId}`,
                    object: "chat.completion.chunk",
                    created: Math.floor(Date.now() / 1000),
                    model: state.model,
                    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
                });
                console.log(`[STREAM] Synthesized finish_reason=${finishReason} (upstream ended without message_stop)`);
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

async function handleChatCompletions(req, res) {
    const requestStart = Date.now();

    // Fast-path: respond instantly to Cursor's model-validation pings
    // (non-streaming, no tools, 1-2 messages) to prevent disconnect during Azure cold start
    if (isModelValidationPing(req.body)) {
        console.log(`[PROXY] Model validation ping (model=${req.body.model}, stream=${req.body.stream}), responding locally`);
        return res.json(makeValidationResponse(req.body.model || DEFAULT_DEPLOYMENT));
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

        const anthropicRequest = buildAnthropicRequest(req.body);

        console.log(`[PROXY] ── Request ──────────────────────────────────`);
        console.log(`[PROXY] cursor_model=${req.body.model} → deployment=${anthropicRequest.model}`);
        console.log(`[PROXY] cursor_max_tokens=${req.body.max_tokens || 'not set'} → actual_max_tokens=${anthropicRequest.max_tokens}`);
        console.log(`[PROXY] stream=${isStreaming}, tools=${anthropicRequest.tools?.length || 0}, messages=${anthropicRequest.messages.length}${anthropicRequest.thinking ? ', thinking=enabled(budget=' + anthropicRequest.thinking.budget_tokens + ')' : ''}`);
        console.log(`[PROXY] Calling Azure endpoint...`);

        // Retry logic for transient errors (429 rate limit, 529 overloaded)
        const MAX_RETRIES = 3;
        let response;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
                headers: {
                    "Content-Type": "application/json",
                    "x-api-key": CONFIG.AZURE_API_KEY,
                    "anthropic-version": CONFIG.ANTHROPIC_VERSION,
                },
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
            handleAnthropicStream(response, res, req.body.model, abortController, true);
        } else {
            const anthropicStopReason = response.data?.stop_reason;
            const openaiResponse = convertAnthropicResponseToOpenai(response.data, req.body.model);
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
    for (const deployment of Object.values(MODEL_MAP)) {
        if (!seen.has(deployment)) {
            seen.add(deployment);
            models.push({ id: deployment, object: "model", created: 1700000000, owned_by: "azure-anthropic" });
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
        const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
            },
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
    console.log(`Default Deployment: ${DEFAULT_DEPLOYMENT}`);
    console.log(`Model Map: ${JSON.stringify(MODEL_MAP)}`);
    console.log(`Endpoint: ${CONFIG.AZURE_ENDPOINT}`);
    console.log(`Thinking Budget: ${THINKING_BUDGET_TOKENS} tokens (env THINKING_BUDGET_TOKENS)`);
    console.log(`Min Output Tokens: ${MIN_OUTPUT_TOKENS}`);
    console.log(`API Key: ${CONFIG.AZURE_API_KEY ? "configured" : "MISSING"}`);
    console.log(`Auth Key: ${CONFIG.SERVICE_API_KEY ? "configured" : "MISSING"}`);
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
