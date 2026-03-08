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

Server yêu cầu các biến môi trường sau:

-   `AZURE_ENDPOINT` - Azure Anthropic API endpoint
-   `AZURE_API_KEY` - Azure API key
-   `SERVICE_API_KEY` - Service API key dùng để xác thực request từ Cursor IDE (phải khớp với API Key trong Cursor Settings)
-   `PORT` - Port để chạy server (mặc định: 3000)
-   `AZURE_DEPLOYMENT_NAME` - Tên deployment trên Azure (mặc định: "claude-opus-4-5")

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
     AZURE_ENDPOINT=https://<resource>.openai.azure.com/anthropic/v1/messages
     AZURE_API_KEY=your-azure-api-key
     SERVICE_API_KEY=your-random-secret-key
     PORT=3000
     AZURE_DEPLOYMENT_NAME=claude-opus-4-5
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
