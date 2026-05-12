alter table public.practice_history
alter column problem_phonemes drop default;

alter table public.practice_history
alter column problem_phonemes type jsonb
using to_jsonb(problem_phonemes);

alter table public.practice_history
alter column problem_phonemes set default '[]'::jsonb;

alter table public.practice_history
alter column problem_phonemes set not null;

alter table public.practice_history
add column if not exists feedback jsonb not null default '{}'::jsonb;

alter table public.practice_history
drop constraint if exists practice_history_problem_phonemes_jsonb_check;

alter table public.practice_history
add constraint practice_history_problem_phonemes_jsonb_check
check (jsonb_typeof(problem_phonemes) = 'array');

alter table public.practice_history
drop constraint if exists practice_history_feedback_jsonb_check;

alter table public.practice_history
add constraint practice_history_feedback_jsonb_check
check (jsonb_typeof(feedback) = 'object');