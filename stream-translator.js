// Claude SSE → OpenAI SSE stream translator
// Based on 9router's claude-to-openai.js (proven reliable with Cursor IDE)

function createChunk(state, delta, finishReason = null) {
    return {
        id: `chatcmpl-${state.messageId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: state.model,
        choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
}

function convertStopReason(reason) {
    switch (reason) {
        case "end_turn": return "stop";
        case "max_tokens": return "length";
        case "tool_use": return "tool_calls";
        case "stop_sequence": return "stop";
        default: return "stop";
    }
}

function createStreamState(model) {
    return {
        messageId: "msg_" + Date.now(),
        model: model,
        toolCallIndex: 0,
        toolCalls: new Map(),
        finishReason: null,
        finishReasonSent: false,
        usage: null,
        inThinkingBlock: false,
        currentBlockIndex: -1,
        currentBlockType: null,
        serverToolBlockIndex: -1,
        lastChunkHash: null,
        duplicateCount: 0,
    };
}

function translateClaudeEvent(event, state) {
    if (!event) return null;

    const results = [];
    const eventType = event.type;

    switch (eventType) {
        case "message_start": {
            state.messageId = event.message?.id || `msg_${Date.now()}`;
            if (event.message?.model) state.model = event.message.model;
            state.toolCallIndex = 0;

            if (event.message?.usage?.input_tokens) {
                state.usage = { input_tokens: event.message.usage.input_tokens, output_tokens: 0 };
            }

            // OpenAI format: first chunk always has role + content:null
            results.push(createChunk(state, { role: "assistant", content: null }));
            break;
        }

        case "content_block_start": {
            const block = event.content_block;
            if (!block) break;

            state.currentBlockType = block.type;
            state.currentBlockIndex = event.index;

            if (block.type === "server_tool_use") {
                state.serverToolBlockIndex = event.index;
                break;
            }

            if (block.type === "thinking" || block.type === "redacted_thinking") {
                state.inThinkingBlock = true;
            } else if (block.type === "tool_use") {
                const toolCallIndex = state.toolCallIndex++;
                const toolCall = {
                    index: toolCallIndex,
                    id: block.id,
                    type: "function",
                    function: { name: block.name, arguments: "" },
                    _hasReceivedArgs: false,
                };
                state.toolCalls.set(event.index, toolCall);
                // First chunk for this tool call includes id, type, and name
                results.push(createChunk(state, {
                    tool_calls: [{
                        index: toolCallIndex,
                        id: block.id,
                        type: "function",
                        function: { name: block.name, arguments: "" },
                    }],
                }));
            }
            break;
        }

        case "content_block_delta": {
            if (event.index === state.serverToolBlockIndex) break;

            const delta = event.delta;
            if (!delta) break;

            if (delta.type === "text_delta" && delta.text) {
                // Dedup protection: skip identical consecutive text chunks
                const hash = delta.text;
                if (hash === state.lastChunkHash) {
                    state.duplicateCount++;
                    if (state.duplicateCount > 3) break;
                } else {
                    state.lastChunkHash = hash;
                    state.duplicateCount = 0;
                }
                results.push(createChunk(state, { content: delta.text }));
            } else if (delta.type === "thinking_delta" && delta.thinking) {
                results.push(createChunk(state, { reasoning_content: delta.thinking }));
            } else if (delta.type === "input_json_delta" && delta.partial_json !== undefined) {
                const toolCall = state.toolCalls.get(event.index);
                if (toolCall) {
                    toolCall.function.arguments += delta.partial_json;
                    toolCall._hasReceivedArgs = true;
                    // Delta chunks: only index + argument fragment (no id/type/name)
                    results.push(createChunk(state, {
                        tool_calls: [{
                            index: toolCall.index,
                            function: { arguments: delta.partial_json },
                        }],
                    }));
                }
            }
            break;
        }

        case "content_block_stop": {
            if (event.index === state.serverToolBlockIndex) {
                state.serverToolBlockIndex = -1;
                state.currentBlockType = null;
                break;
            }
            // Emit valid empty args for parameterless tool calls (LiteLLM pattern)
            const stoppedToolCall = state.toolCalls.get(event.index);
            if (stoppedToolCall && !stoppedToolCall._hasReceivedArgs) {
                stoppedToolCall.function.arguments = "{}";
                results.push(createChunk(state, {
                    tool_calls: [{
                        index: stoppedToolCall.index,
                        function: { arguments: "{}" },
                    }],
                }));
            }
            if (state.inThinkingBlock && event.index === state.currentBlockIndex) {
                state.inThinkingBlock = false;
            }
            state.currentBlockType = null;
            break;
        }

        case "message_delta": {
            if (event.usage && typeof event.usage === "object") {
                const inputTokens = state.usage?.input_tokens || 0;
                const outputTokens = typeof event.usage.output_tokens === "number" ? event.usage.output_tokens : 0;
                const cacheRead = typeof event.usage.cache_read_input_tokens === "number" ? event.usage.cache_read_input_tokens : 0;
                const cacheCreation = typeof event.usage.cache_creation_input_tokens === "number" ? event.usage.cache_creation_input_tokens : 0;

                state.usage = { input_tokens: inputTokens, output_tokens: outputTokens };
                if (cacheRead > 0) state.usage.cache_read_input_tokens = cacheRead;
                if (cacheCreation > 0) state.usage.cache_creation_input_tokens = cacheCreation;
            }

            if (event.delta?.stop_reason) {
                const rawStopReason = event.delta.stop_reason;
                state.finishReason = convertStopReason(rawStopReason);
                // Override: if we emitted tool calls but stop_reason is "end_turn",
                // Cursor needs "tool_calls" to trigger tool execution
                if (state.finishReason === "stop" && state.toolCalls.size > 0) {
                    state.finishReason = "tool_calls";
                }

                const outputTokens = event.usage?.output_tokens || state.usage?.output_tokens || '?';
                console.log(`[STREAM] ⚡ stop_reason=${rawStopReason} → finish_reason=${state.finishReason}, output_tokens=${outputTokens}, tool_calls=${state.toolCalls.size}`);
                if (rawStopReason === "max_tokens") {
                    console.log(`[STREAM] ⚠️  OUTPUT TRUNCATED — model hit max_tokens limit. Increase max_tokens or check resolveMaxTokens().`);
                }

                const finalChunk = {
                    id: `chatcmpl-${state.messageId}`,
                    object: "chat.completion.chunk",
                    created: Math.floor(Date.now() / 1000),
                    model: state.model,
                    choices: [{ index: 0, delta: {}, finish_reason: state.finishReason }],
                };

                if (state.usage) {
                    const input = state.usage.input_tokens || 0;
                    const output = state.usage.output_tokens || 0;
                    const cached = state.usage.cache_read_input_tokens || 0;
                    const cacheCreation = state.usage.cache_creation_input_tokens || 0;
                    const promptTokens = input + cached + cacheCreation;
                    finalChunk.usage = {
                        prompt_tokens: promptTokens,
                        completion_tokens: output,
                        total_tokens: promptTokens + output,
                    };
                }

                results.push(finalChunk);
                state.finishReasonSent = true;
            }
            break;
        }

        case "message_stop": {
            if (!state.finishReasonSent) {
                const finishReason = state.finishReason || (state.toolCalls.size > 0 ? "tool_calls" : "stop");
                const finalChunk = {
                    id: `chatcmpl-${state.messageId}`,
                    object: "chat.completion.chunk",
                    created: Math.floor(Date.now() / 1000),
                    model: state.model,
                    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
                };
                if (state.usage) {
                    const input = state.usage.input_tokens || 0;
                    const output = state.usage.output_tokens || 0;
                    finalChunk.usage = {
                        prompt_tokens: input,
                        completion_tokens: output,
                        total_tokens: input + output,
                    };
                }
                results.push(finalChunk);
                state.finishReasonSent = true;
            }
            break;
        }

        case "ping":
            break;
    }

    return results.length > 0 ? results : null;
}

module.exports = { createStreamState, translateClaudeEvent, convertStopReason };
