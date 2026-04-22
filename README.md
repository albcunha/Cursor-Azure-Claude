# Azure Anthropic Proxy for Cursor

Proxy server để kết nối Cursor IDE với Azure Anthropic API (Claude).

## 🌐 Production URLs

-   **Base URL**: https://cursor-azure-claude-proxy-production.up.railway.app/
-   **Health Check**: https://cursor-azure-claude-proxy-production.up.railway.app/health

## 📋 Endpoints

### Root Endpoint

-   `GET /` - Thông tin về server và các endpoints có sẵn

### Health Check

-   `GET /health` - Kiểm tra trạng thái server

### Chat Endpoints

-   `POST /chat/completions` - Endpoint chính cho Cursor IDE (OpenAI format)
-   `POST /v1/chat/completions` - OpenAI format
-   `POST /v1/messages` - Anthropic native format

## 🚀 Cách sử dụng

### Cấu hình trong Cursor IDE

1. Mở Cursor Settings
2. Tìm phần "Model" hoặc "Model Settings" Mở "Opus 4.5"
3. API Keys mucj OpenAI Custom API URL: `https://cursor-azure-claude-proxy-production.up.railway.app`
4. Đặt API Key: Giá trị phải **trùng khớp chính xác** với biến môi trường `SERVICE_API_KEY` trong file `.env` của server. Bật OpenAI API key

![Cấu hình Model trong Cursor IDE](screenshot/cursor-model.png)

![Cấu hình Chat trong Cursor IDE](screenshot/cursor-chat.png)

**Lưu ý quan trọng**: API Key trong Cursor IDE (`Cursor Settings > Models > API Keys > OpenAI API Key`) phải khớp chính xác với giá trị `SERVICE_API_KEY` trong file `.env` của server. Nếu không khớp, request sẽ bị từ chối với lỗi authentication.

### Ví dụ Request

```bash
curl -X POST https://cursor-azure-claude-proxy-production.up.railway.app/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SERVICE_API_KEY" \
  -d '{
    "model": "claude-opus-4-5",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Lưu ý**: Thay `YOUR_SERVICE_API_KEY` bằng giá trị thực từ biến môi trường `SERVICE_API_KEY`.

## ⚙️ Environment Variables

### Claude (Azure Anthropic)

-   `AZURE_ENDPOINT` - Azure Anthropic API endpoint (`https://<resource>.services.ai.azure.com/anthropic/v1/messages`). The GPT base URL is derived from this automatically.
-   `AZURE_API_KEY` - Azure API key (shared with the OpenAI endpoint on the same Foundry resource)
-   `AZURE_CLAUDE_DEPLOYMENT_NAME` - Default Claude deployment name, e.g. `claude-opus-4-7`. Used for every Claude request unless a family-specific override is set.
-   *(optional)* `AZURE_CLAUDE_OPUS_DEPLOYMENT`, `AZURE_CLAUDE_SONNET_DEPLOYMENT`, `AZURE_CLAUDE_HAIKU_DEPLOYMENT` - override the deployment per model family

### GPT-5.4 (Azure OpenAI)

Leave these unset to disable the GPT route; Claude keeps working.

-   `AZURE_OPENAI_API_VERSION` - Default `2025-04-01-preview` (supports gpt-5.4 chat completions + `reasoning_effort`)
-   `AZURE_GPT_DEPLOYMENT` - Exact name of the gpt-5.4 deployment in your Azure resource. Default `gpt-5.4`
-   `GPT_REASONING_EFFORT` - Fallback effort (`minimal` / `low` / `medium` / `high`) when Cursor sends the bare `gpt-5.4` id without a suffix. Default `medium`
-   *(optional)* `AZURE_OPENAI_ENDPOINT` - only if your GPT deployment lives on a different resource than Claude; defaults to the host of `AZURE_ENDPOINT`
-   *(optional)* `AZURE_OPENAI_API_KEY` - defaults to `AZURE_API_KEY`

### Service

-   `SERVICE_API_KEY` - Key used to authenticate Cursor against this proxy (must match Cursor's **OpenAI API Key** field)
-   `PORT` - Server port (default 8080; Railway assigns its own)
-   *(optional)* `LOG_TOOL_CALLS`, `LOG_MESSAGES` - verbose logging toggles (`true`/`false`)

## 🤖 Using GPT-5.4 in Cursor

Azure Foundry ships `gpt-5.4` with native Chat Completions support since 2026-03-05, so the proxy forwards Cursor's OpenAI request almost unchanged (only swapping the model name to the deployment, stripping `temperature`/`top_p`, and converting `max_tokens` → `max_completion_tokens`, as required by reasoning models).

1. Deploy `gpt-5.4` in the same Foundry resource you use for Claude. Note that `gpt-5.4` and `gpt-5.4-pro` currently require [access registration](https://aka.ms/OAI/gpt53codexaccess) on Azure.
2. Set the `AZURE_OPENAI_*` env vars above on Railway and redeploy.
3. In **Cursor Settings → Models → Custom Models**, add any of these model ids. Each one maps to the same Azure deployment, only `reasoning_effort` changes:

    | Cursor model id   | reasoning_effort |
    | ----------------- | ---------------- |
    | `gpt-5.4`         | falls back to `GPT_REASONING_EFFORT` (default `medium`) |
    | `gpt-5.4-minimal` | `minimal`        |
    | `gpt-5.4-low`     | `low`            |
    | `gpt-5.4-medium`  | `medium`         |
    | `gpt-5.4-high`    | `high`           |

4. Keep the **OpenAI API Key** field set to your `SERVICE_API_KEY` and the base URL pointing at this proxy — both Claude and GPT-5.4 share the same proxy endpoint and auth key.

### Quick smoke test

```bash
curl -X POST https://cursor-azure-claude-proxy-production.up.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{
    "model": "gpt-5.4-medium",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }'
```

## 📦 Installation

```bash
npm install
npm start
```

## 🔧 Development

```bash
npm run dev
```

## 🚂 Deploy trên Railway

### Cấu hình nhanh

1. **Tạo project mới trên Railway**
   - Truy cập [Railway](https://railway.app)
   - Tạo project mới từ GitHub repository hoặc Deploy từ GitHub

2. **Cấu hình Environment Variables**
   - Vào tab **Variables** trong Railway project
   - Thêm các biến môi trường sau:
     ```
     # Azure Anthropic
     AZURE_ENDPOINT=https://<resource>.services.ai.azure.com/anthropic/v1/messages
     AZURE_API_KEY=your-azure-api-key
     AZURE_CLAUDE_DEPLOYMENT_NAME=claude-opus-4-7

     # Azure OpenAI (gpt-5.4)
     AZURE_OPENAI_API_VERSION=2025-04-01-preview
     AZURE_GPT_DEPLOYMENT=gpt-5.4
     GPT_REASONING_EFFORT=high

     # Service
     SERVICE_API_KEY=your-random-secret-key
     PORT=3000
     LOG_TOOL_CALLS=false
     LOG_MESSAGES=false
     ```
   - **Lưu ý**: `SERVICE_API_KEY` để bảo vệ dịch vụ của bạn. Hãy đặt nó thành một chuỗi ký tự ngẫu nhiên.

   ![Cấu hình Environment Variables trên Railway](screenshot/railway-var.png)

3. **Cấu hình Build Settings**
   - Railway sẽ tự động detect Node.js project

4. **Deploy**
   - Railway sẽ tự động deploy khi bạn push code lên GitHub
   - Hoặc click **Deploy** trong Railway dashboard
   - Sau khi deploy thành công, Railway sẽ cung cấp một public URL

5. **Kiểm tra Health Check**
   - Truy cập: `https://your-app.up.railway.app/health`
   - Nếu trả về `{"status":"ok"}`, server đã chạy thành công

6. **Cấu hình Custom Domain (tùy chọn)**
   - Vào tab **Settings** > **Networking**
   - Thêm custom domain nếu cần

   ![Cấu hình Custom Domain trên Railway](screenshot/railway-domain.png)

### Lưu ý khi deploy

- Railway tự động cung cấp `PORT` qua biến môi trường, nhưng bạn vẫn có thể set `PORT=8080` để đảm bảo
- `SERVICE_API_KEY` phải khớp chính xác với API Key bạn cấu hình trong Cursor IDE
- Kiểm tra logs trong Railway dashboard nếu gặp lỗi

## 📝 License

MIT

## 🙏 Tham khảo

Dự án này được tham khảo từ [Cursor-Azure-GPT-5](https://github.com/gabrii/Cursor-Azure-GPT-5) - một service cho phép Cursor sử dụng Azure GPT-5 deployments.
