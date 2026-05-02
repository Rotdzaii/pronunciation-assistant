# README – Phần Backend + Database hiện tại của Triết

## 1. Mục đích tài liệu

Tài liệu này dùng để bàn giao **phần backend + database** mà Triết đã làm cho đề tài:

**“Nghiên cứu và phát triển hệ thống chẩn đoán lỗi phát âm tiếng Anh tự động dựa trên Deep Learning”**

README này giúp nhóm trưởng hoặc thành viên khác:
- hiểu phần nào đã làm xong
- biết cách khởi động môi trường
- biết cách chạy backend hiện tại
- biết cách test các API đã có
- biết phần nào chưa làm xong để tiếp tục

---

## 2. Phạm vi Triết phụ trách

Triết phụ trách chính:
- **Database**
- **Backend (NestJS / Node.js)**
- **Supabase integration**
- **Auth & RBAC**
- **Audio upload + Storage**
- **Async job flow cơ bản**
- **Polling job status**
- **Webhook nhận kết quả AI**
- **Queue setup bước đầu bằng Supabase pgmq**

---

## 3. Kiến trúc tổng thể hiện tại

Hệ thống hiện tại đi theo hướng microservices:

- **Client / Frontend / App**
- **Node.js API Gateway (NestJS)** ← phần Triết đang làm
- **FastAPI AI Service** ← service AI xử lý phát âm
- **Supabase**
  - Auth
  - Postgres
  - Storage
- **Queue**: Supabase `pgmq` với queue `practice_jobs`

### Flow tổng quát

1. Student đăng nhập bằng Supabase Auth
2. Student upload audio
3. Backend validate file
4. Backend upload audio lên Supabase Storage
5. Backend tạo job trong `practice_history`
6. Backend trả `job_id` cho client
7. Client polling theo `job_id`
8. AI/FastAPI xử lý xong sẽ gọi webhook về backend
9. Backend cập nhật job thành `completed` hoặc `failed`
10. Frontend đọc lại trạng thái mới

> **Lưu ý:** Queue `practice_jobs` đã được tạo, nhưng phần **enqueue thật vào queue từ backend** vẫn là bước tiếp theo, chưa nối hoàn chỉnh trong flow hiện tại.

---

## 4. Tech stack đang dùng

### Backend
- Node.js
- NestJS
- TypeScript

### Database / Auth / Storage
- Supabase
  - Authentication
  - PostgreSQL
  - Storage

### AI service
- FastAPI (Python) – phía service AI

### Queue
- Supabase `pgmq`
- Queue name: `practice_jobs`

### Tools hỗ trợ
- Postman
- Docker Desktop
- FFmpeg

---

## 5. Cấu trúc backend hiện tại

Thư mục backend đang theo kiểu một project NestJS duy nhất, tách theo module:

```text
backend/
├── src/
│   ├── auth/
│   ├── practice/
│   ├── supabase/
│   ├── app.controller.ts
│   ├── app.module.ts
│   └── main.ts
├── .env
├── package.json
├── tsconfig.json
├── tsconfig.build.json
└── nest-cli.json
```

### Ý nghĩa từng phần

#### `src/auth/`
Chứa phần:
- xác thực JWT bằng Supabase Auth
- RBAC `student/teacher`
- route test quyền truy cập

#### `src/practice/`
Chứa phần flow chính:
- upload audio
- create job
- polling job status
- webhook nhận kết quả AI

#### `src/supabase/`
Chứa service kết nối Supabase:
- client thường
- admin client

---

## 6. Những gì đã làm xong

### 6.1. Auth & RBAC
Đã làm xong:
- xác thực JWT bằng Supabase
- route public `/health`
- route protected `/auth/me`
- role nghiệp vụ bằng bảng `profiles`
- role test:
  - student vào route student → OK
  - teacher vào route teacher → OK
  - sai quyền → `403 Forbidden`

### 6.2. Upload & Storage
Đã làm xong:
- validate audio upload
- chỉ chấp nhận:
  - `audio/wav`
  - `audio/mpeg`
  - `audio/mp4`
- giới hạn 5MB
- upload file lên Supabase Storage bucket `practice-audios`
- trả về `audio_url`

### 6.3. Async job flow cơ bản
Đã làm xong:
- `POST /practice/create-job`
- tạo bản ghi trong `practice_history`
- trạng thái ban đầu `processing`
- trả `job_id`

### 6.4. Polling
Đã làm xong:
- `GET /practice/:job_id`
- student chỉ xem được job của chính mình
- trả trạng thái hiện tại của job

### 6.5. Webhook nhận kết quả AI
Đã làm xong dạng test/manual:
- `POST /practice/webhook/ai-result`
- cập nhật `practice_history`
- đổi sang `completed` hoặc `failed`
- lưu `score` và `problem_phonemes`

### 6.6. Queue bước đầu
Đã làm xong:
- bật extension `pgmq`
- tạo queue `practice_jobs`
- chốt payload queue gồm 4 trường:
  - `job_id`
  - `student_id`
  - `target_word`
  - `audio_url`

---

## 7. Những gì chưa làm xong

Các phần sau **chưa hoàn tất hoàn toàn**:

1. **Enqueue thật từ backend vào queue `practice_jobs`**
   - hiện tại `create-job` mới insert DB
   - chưa push message thật vào queue trong code backend

2. **Worker / consumer đọc queue thật**
   - chưa có flow hoàn chỉnh để FastAPI tự đọc `practice_jobs`

3. **Bảo mật webhook nội bộ**
   - hiện webhook đang để dễ test
   - nên thêm secret/token nội bộ sau

4. **Teacher analytics hoàn chỉnh**
5. **Swagger/OpenAPI hoàn chỉnh**
6. **Rate limiting / centralized logging hoàn chỉnh**

---

## 8. Database hiện tại

### 8.1. Bảng `profiles`
Dùng để lưu role nghiệp vụ.

Các cột chính:
- `id` – trùng với `auth.users.id`
- `email`
- `role` – `student` hoặc `teacher`
- `created_at`
- `updated_at`

### 8.2. Bảng `practice_history`
Dùng để lưu job chấm điểm phát âm.

Các cột chính:
- `id` – `job_id`
- `student_id`
- `target_word`
- `audio_url`
- `status` – `processing | completed | failed`
- `score`
- `problem_phonemes` – `jsonb`
- `created_at`
- `updated_at`

---

## 9. Supabase hiện tại

### Đã dùng các phần sau:
- **Authentication**
- **SQL Editor / PostgreSQL**
- **Storage**
- **pgmq**

### Bucket hiện tại
```text
practice-audios
```

### Queue hiện tại
```text
practice_jobs
```

---

## 10. Biến môi trường cần có

Trong file `.env` của backend:

```env
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
PORT=3000
```

> Không commit `.env` lên Git.

---

## 11. Cách khởi động phần Triết đã làm

### Bước 1: mở Docker Desktop
- Mở Docker Desktop
- Chờ tới khi thấy `Engine running`

> Nếu chỉ dùng Supabase cloud dashboard thì Docker không phải lúc nào cũng bắt buộc cho mọi bước test API, nhưng nên mở sẵn để tránh thiếu runtime/tooling local.

### Bước 2: mở project backend
Đi tới thư mục backend:

```powershell
cd D:\School\NCKH\pronunciation-assistant\backend
```

### Bước 3: cài package nếu máy mới chưa cài

```powershell
npm install
```

### Bước 4: chạy backend

```powershell
npm run start:dev
```

### Kết quả mong muốn
Terminal sẽ hiện kiểu:
- `Nest application successfully started`
- map các route `/health`, `/auth/me`, `/practice/...`

### Bước 5: test nhanh server
Dùng Postman gọi:

```http
GET http://localhost:3000/health
```

Nếu đúng sẽ trả:

```json
{
  "message": "ok"
}
```

---

## 12. Postman setup hiện tại

### Environment `local`
Các biến đang dùng:

- `base_url`
- `supabase_url`
- `supabase_anon_key`
- `student_email`
- `student_password`
- `teacher_email`
- `teacher_password`
- `student_access_token`
- `teacher_access_token`

### Quy tắc dùng token
- gọi Supabase Auth API → dùng `apikey = supabase_anon_key`
- route backend của student → `Bearer {{student_access_token}}`
- route backend của teacher → `Bearer {{teacher_access_token}}`

### Lưu ý
Mỗi lần mở máy lại hoặc sau một thời gian dài, nên login lại để cập nhật:
- `student_access_token`
- `teacher_access_token`

---

## 13. Các API hiện tại đã test được

### 13.1. Health check
```http
GET /health
```
- public
- trả `{ "message": "ok" }`

### 13.2. Auth me
```http
GET /auth/me
```
- protected
- dùng JWT hợp lệ
- trả:
  - `id`
  - `email`
  - `auth_role`
  - `app_role`

### 13.3. Student test route
```http
GET /auth/student/test
```
- student → `200`
- teacher → `403`

### 13.4. Teacher test route
```http
GET /auth/teacher/test
```
- teacher → `200`
- student → `403`

### 13.5. Upload audio
```http
POST /practice/upload-audio
```
- chỉ student
- multipart/form-data
- validate MIME + size
- upload lên Storage
- trả `audio_url`

### 13.6. Create job
```http
POST /practice/create-job
```
- chỉ student
- body gồm:
  - `target_word`
  - `audio_url`
- tạo record trong `practice_history`
- trả:
  - `job_id`
  - `status = processing`

### 13.7. Polling job status
```http
GET /practice/:job_id
```
- chỉ student
- chỉ xem job của chính mình
- trả trạng thái job hiện tại

### 13.8. Webhook AI result
```http
POST /practice/webhook/ai-result
```
- hiện đang dùng để test/manual callback
- body ví dụ:

```json
{
  "job_id": "...",
  "status": "completed",
  "score": 85,
  "problem_phonemes": ["/k/", "/juː/"]
}
```

---

## 14. Flow hiện tại của backend

### Flow student luyện phát âm
1. Student login
2. Student upload audio
3. Backend validate file
4. Backend upload file lên Supabase Storage
5. Backend trả `audio_url`
6. Frontend gọi `create-job` với `target_word + audio_url`
7. Backend tạo job trong `practice_history`
8. Frontend polling bằng `job_id`
9. AI/FastAPI (hoặc test manual) gọi webhook
10. Backend cập nhật job completed/failed
11. Frontend polling lại để đọc kết quả mới

---

## 15. Queue hiện tại

### Đã có:
- extension `pgmq`
- queue `practice_jobs`

### Payload queue đã chốt:

```json
{
  "job_id": "uuid",
  "student_id": "uuid",
  "target_word": "computer",
  "audio_url": "https://..."
}
```

### Ý định của bước tiếp theo
Sau khi `POST /practice/create-job` insert vào `practice_history`, backend sẽ **enqueue message vào `practice_jobs`**.

Hiện tại bước này **chưa hoàn tất trong code**.

---

## 16. Git / branch hiện tại

Branch đã push phần hiện tại của backend:

```text
triet/feature/practice-flow
```

Ý định branch tiếp theo:

```text
triet/feature/queue-pgmq
```

---

## 17. Việc tiếp theo nhóm trưởng hoặc người tiếp tục nên làm

Ưu tiên tiếp theo nên là:

### 17.1. Nối queue vào create-job
- sau khi insert `practice_history`
- push payload vào `practice_jobs`

### 17.2. Làm worker/consumer
- để FastAPI đọc queue thật
- xử lý job thật

### 17.3. Bảo mật webhook
- thêm secret nội bộ

### 17.4. Hoàn thiện analytics và docs
- teacher analytics
- Swagger/OpenAPI
- logging
- rate limiting

---

## 18. Ghi chú quan trọng

- Code hiện tại là **một backend NestJS duy nhất**, tách theo module `auth`, `practice`, `supabase`.
- Không tách thành nhiều project con.
- Feature hiện tại đã được gom và push thành một nhánh lớn `triet/feature/practice-flow`.
- Từ bước tiếp theo trở đi nên làm theo từng feature branch riêng.

---

## 19. Tóm tắt siêu ngắn

Triết đã làm được:
- JWT auth
- RBAC
- upload audio
- upload Storage
- create job
- polling
- webhook update result
- tạo queue `practice_jobs`

Triết chưa làm xong:
- enqueue thật vào queue trong code backend
- consumer/worker đọc queue thật
- bảo mật webhook
- analytics/docs hoàn chỉnh
