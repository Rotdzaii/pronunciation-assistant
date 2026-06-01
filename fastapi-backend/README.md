# Phần Máy Chủ FastAPI Cho Pronunciation Assistant

Phần máy chủ FastAPI cho bản thử nghiệm luyện phát âm trên Expo. Phần máy chủ xác thực mã truy cập của Supabase, lấy vai trò ứng dụng từ bảng `public.profiles`, tải âm thanh luyện tập lên Supabase Storage, tạo bài luyện tập và nhận kết quả mô phỏng từ AI qua móc nối webhook dùng khóa bí mật chung.

## Cài đặt

```powershell
cd fastapi-backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
copy .env.example .env
```

Điền các giá trị Supabase sau vào file `.env`:

```dotenv
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_ANON_KEY="your-supabase-anon-key"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
SUPABASE_JWT_SECRET="your-supabase-jwt-secret"
AI_WEBHOOK_SECRET="replace-with-ai-webhook-secret"
PRACTICE_AUDIO_BUCKET="practice-audios"
```

Chạy API:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mở các địa chỉ sau:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Các API Hiện Có

- `GET /health`
- `GET /auth/me`
- `POST /practice/upload-audio`
- `POST /practice/create-job`
- `GET /practice/{job_id}`
- `POST /practice/webhook/ai-result`
- `GET /practice/history`

## Xác thực

`GET /auth/me` cần mã truy cập của Supabase:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" http://localhost:8000/auth/me
```

Phản hồi thành công:

```json
{
  "id": "user-id",
  "email": "user@example.com",
  "auth_role": "authenticated",
  "app_role": "student"
}
```

Điểm cuối này xác thực mã truy cập của Supabase bằng điểm cuối `/auth/v1/user` của Supabase Auth, sau đó lấy `app_role` từ bảng `profiles` bằng khóa vai trò dịch vụ.

## Tải Âm Thanh

`POST /practice/upload-audio` cần mã truy cập của Supabase cho người dùng có hồ sơ với `app_role = "student"`.

Bucket Supabase Storage sau phải tồn tại:

```text
practice-audios
```

Tải một tệp âm thanh cục bộ:

```powershell
curl.exe -X POST `
  -H "Authorization: Bearer <supabase-access-token>" `
  -F "file=@C:\path\to\audio.wav;type=audio/wav" `
  http://localhost:8000/practice/upload-audio
```

Với bản ghi từ Expo Web hoặc trình duyệt, file thường có MIME `audio/webm` hoặc `audio/webm;codecs=opus`. Backend sẽ chuẩn hóa giá trị này thành `audio/webm` trước khi kiểm tra.

Các loại MIME được chấp nhận:

- `audio/wav`
- `audio/mpeg`
- `audio/mp4`
- `audio/x-m4a`
- `audio/m4a`
- `audio/webm`
- `audio/ogg`

Phản hồi thành công:

```json
{
  "message": "uploaded",
  "storage_path": "student-id/uuid-audio.wav",
  "audio_url": "https://...",
  "mime_type": "audio/wav",
  "size": 12345
}
```

## Bài Luyện Tập

`POST /practice/create-job` cần mã truy cập của Supabase cho người dùng có hồ sơ với `app_role = "student"`.

Tạo và đưa một bài luyện tập vào hàng đợi sau khi tải âm thanh:

```powershell
curl.exe -X POST `
  -H "Authorization: Bearer <student-supabase-access-token>" `
  -H "Content-Type: application/json" `
  -d "{\"target_word\":\"Architecture\",\"audio_url\":\"https://...\"}" `
  http://localhost:8000/practice/create-job
```

Phản hồi thành công:

```json
{
  "job_id": "practice-job-id",
  "status": "processing",
  "message": "Practice job created and queued"
}
```

API ghi dữ liệu vào `public.practice_history` với `problem_phonemes = []` và `feedback = {}`, sau đó gọi `public.enqueue_practice_job(...)`.

Lấy thông tin một bài luyện tập:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" `
  http://localhost:8000/practice/<practice-job-id>
```

Học viên chỉ có thể xem bài luyện tập của chính mình. Giáo viên có thể xem mọi bài luyện tập khi hồ sơ có `app_role = "teacher"`.

Liệt kê lịch sử luyện tập với vai trò học viên:

```powershell
curl.exe -H "Authorization: Bearer <student-supabase-access-token>" `
  "http://localhost:8000/practice/history?limit=20&offset=0"
```

Liệt kê các bài đã hoàn thành với vai trò giáo viên cho một học viên:

```powershell
curl.exe -H "Authorization: Bearer <teacher-supabase-access-token>" `
  "http://localhost:8000/practice/history?student_id=<student-id>&status=completed&limit=20&offset=0"
```

Phản hồi lịch sử thành công:

```json
{
  "items": [
    {
      "id": "practice-job-id",
      "student_id": "student-id",
      "target_word": "Architecture",
      "audio_url": "https://...",
      "status": "completed",
      "score": 86.5,
      "problem_phonemes": [],
      "feedback": {},
      "created_at": "2026-05-12T00:00:00Z",
      "updated_at": "2026-05-12T00:01:00Z"
    }
  ],
  "limit": 20,
  "offset": 0
}
```

## Móc Nối Webhook Kết Quả AI

`POST /practice/webhook/ai-result` dành cho tiến trình AI. Điểm cuối này không dùng JWT của người dùng. Điểm cuối yêu cầu tiêu đề HTTP chứa khóa bí mật chung `x-ai-webhook-secret`.

Đánh dấu một bài luyện tập là hoàn thành:

```powershell
curl.exe -X POST `
  -H "x-ai-webhook-secret: <ai-webhook-secret>" `
  -H "Content-Type: application/json" `
  -d "{\"job_id\":\"<practice-job-id>\",\"status\":\"completed\",\"score\":82.5,\"problem_phonemes\":[],\"feedback\":{}}" `
  http://localhost:8000/practice/webhook/ai-result
```

Đánh dấu một bài luyện tập là thất bại:

```powershell
curl.exe -X POST `
  -H "x-ai-webhook-secret: <ai-webhook-secret>" `
  -H "Content-Type: application/json" `
  -d "{\"job_id\":\"<practice-job-id>\",\"status\":\"failed\",\"problem_phonemes\":[],\"feedback\":{}}" `
  http://localhost:8000/practice/webhook/ai-result
```

Phản hồi thành công:

```json
{
  "job_id": "practice-job-id",
  "status": "completed",
  "message": "Practice job result updated"
}
```
