# Demo Data

Tài liệu này dùng để seed dữ liệu demo cho Pronunciation Assistant / Phoenix.

## Biến môi trường

Đặt trong `fastapi-backend/.env` hoặc export trực tiếp trong terminal:

```env
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
DEMO_DEFAULT_PASSWORD=...
# hoặc dùng biến cũ nếu môi trường hiện tại đã có:
# DEMO_PASSWORD=...
BASE_URL=http://127.0.0.1:8000
```

Không commit `.env`, `.tokens.local.env`, service role key, JWT hoặc password thật.

## Chạy seed

Từ thư mục `fastapi-backend`:

```bash
python scripts/seed_demo_data.py
```

Script dùng Supabase Auth Admin API để tạo user, không insert trực tiếp vào `auth.users`.
Script idempotent: chạy lại sẽ cập nhật user/profile/class/membership/practice history demo thay vì tạo trùng.

## Tài khoản demo

Password lấy từ `DEMO_DEFAULT_PASSWORD`, fallback sang `DEMO_PASSWORD` nếu môi trường cũ đang dùng biến này.

Admin:

- `admin@phoenix-demo.local`

Teacher:

- `teacher01@phoenix-demo.local`
- `teacher02@phoenix-demo.local`
- `teacher03@phoenix-demo.local`

Student:

- `student01@phoenix-demo.local`
- ...
- `student30@phoenix-demo.local`

## Dữ liệu được tạo

- 3 lớp: `DEMO-PHOENIX-A`, `DEMO-PHOENIX-B`, `DEMO-PHOENIX-C`
- 30 học sinh demo, mỗi lớp 10 học sinh
- 3 giáo viên demo
- 1 admin demo
- `DEMO-PHOENIX-B` có 2 giáo viên để test dạy thay
- `teacher_classes.teacher_role` dùng giá trị hợp lệ của schema hiện tại: `owner` và `substitute`
- Mỗi học sinh có một số `practice_history` mẫu dùng cột `score`

## Chạy smoke test

Khởi động backend trước:

```bash
uvicorn app.main:app --reload
```

Sau đó chạy:

```bash
python scripts/smoke_test_demo_data.py
```

Smoke test kiểm tra:

- student/teacher/admin login được
- student xem lớp của mình
- student không xem lớp ngoài phạm vi
- teacher xem lớp mình phụ trách
- teacher xem danh sách học sinh và điểm lớp
- teacher không xem lớp không phụ trách
- admin xem danh sách user/class và demo readiness

## Lưu ý

- Role source chính là `public.profiles.app_role`.
- Teacher-student assignment dùng `student_classes` và `teacher_classes`.
- Không dùng group table.
- Không commit service role key hoặc token sinh ra trong quá trình test.
