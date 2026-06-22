do $$
declare
  inserted_count integer := 0;
begin
  if to_regclass('public.review_requests') is null
     or to_regclass('public.review_audit_logs') is null then
    raise notice 'review request tables are missing. Apply migration 007_create_review_requests.sql first.';
    return;
  end if;

  with candidate_practice as (
    select
      ph.id as practice_history_id,
      ph.student_id,
      sc.class_id,
      row_number() over (order by ph.created_at desc nulls last) as row_number
    from public.practice_history ph
    left join public.student_classes sc
      on sc.student_id = ph.student_id
     and sc.status = 'active'
    where ph.status = 'completed'
    limit 3
  ),
  inserted as (
    insert into public.review_requests (
      practice_history_id,
      student_id,
      class_id,
      source,
      reason,
      student_note,
      severity,
      status
    )
    select
      practice_history_id,
      student_id,
      class_id,
      'student_report',
      case when row_number = 1 then 'ai_scored_wrong'
           when row_number = 2 then 'audio_issue'
           else 'result_mismatch'
      end,
      case when row_number = 1 then 'Demo: học sinh yêu cầu xem lại điểm AI.'
           when row_number = 2 then 'Demo: âm thanh có thể bị lỗi.'
           else 'Demo: kết quả không khớp với bài luyện.'
      end,
      'medium',
      'pending'
    from candidate_practice
    where not exists (
      select 1
      from public.review_requests rr
      where rr.practice_history_id = candidate_practice.practice_history_id
        and rr.source = 'student_report'
        and rr.status in ('pending', 'in_review', 'reanalyzing')
    )
    returning id, student_id, reason, student_note, status
  )
  insert into public.review_audit_logs (
    review_request_id,
    actor_id,
    actor_role,
    action_type,
    after_value
  )
  select
    id,
    student_id,
    'student',
    'student_reported_result',
    jsonb_build_object(
      'reason', reason,
      'student_note', student_note,
      'status', status
    )
  from inserted;

  get diagnostics inserted_count = row_count;
  if inserted_count = 0 then
    raise notice 'No demo review requests inserted. There may be no completed practice_history rows, or requests already exist.';
  else
    raise notice 'Inserted % demo review audit logs and matching review requests.', inserted_count;
  end if;
end $$;
