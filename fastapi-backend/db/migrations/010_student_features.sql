-- Student vocabulary mastery tracking and daily practice goals

CREATE TABLE IF NOT EXISTS student_vocabulary_mastery (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id  uuid        NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    item_id     uuid        NOT NULL REFERENCES vocabulary_items(id) ON DELETE CASCADE,
    mastered_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (student_id, item_id)
);

CREATE INDEX IF NOT EXISTS student_vocabulary_mastery_student_id_idx
    ON student_vocabulary_mastery(student_id);

CREATE TABLE IF NOT EXISTS student_goals (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id   uuid        NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    daily_target int         NOT NULL DEFAULT 5 CHECK (daily_target BETWEEN 1 AND 100),
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (student_id)
);
