-- Phoenix demo cleanup for teacher memberships.
--
-- Purpose:
--   Keep only the owner teacher membership active by default for demo classes.
--   co_teacher/substitute rows should be created only after an invitation is accepted.
--
-- Review before running. This script deletes active co_teacher/substitute rows
-- from teacher_classes for PHOENIX-A/B/C only. Pending invitations remain in
-- teacher_class_invitations.

delete from public.teacher_classes tc
using public.classes c
where tc.class_id = c.id
  and c.code in ('DEMO-PHOENIX-A', 'DEMO-PHOENIX-B', 'DEMO-PHOENIX-C')
  and tc.status = 'active'
  and tc.teacher_role in ('co_teacher', 'substitute');

