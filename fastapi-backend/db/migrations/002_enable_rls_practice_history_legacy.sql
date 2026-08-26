-- practice_history_legacy was deprecated and removed on 2026-07-02.
-- The table was backed up to practice_history_legacy_backup before removal.
-- This guard keeps the migration idempotent on fresh deployments where the table no longer exists.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = 'practice_history_legacy'
  ) THEN
    ALTER TABLE public.practice_history_legacy ENABLE ROW LEVEL SECURITY;
  END IF;
END;
$$;
