# Project Rules

## Database
- NEVER create tables directly in Supabase UI
- Always write migration to db/migrations/ first
- Migration order: check FK deps before numbering
- profiles table PK: id (uuid), no column named teacher_id/student_id