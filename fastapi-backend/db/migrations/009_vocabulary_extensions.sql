-- Add ownership tracking to vocabulary tables and create assignment table

ALTER TABLE vocabulary_sets
    ADD COLUMN IF NOT EXISTS created_by uuid REFERENCES profiles(id);

ALTER TABLE vocabulary_items
    ADD COLUMN IF NOT EXISTS created_by uuid REFERENCES profiles(id);

CREATE TABLE IF NOT EXISTS vocabulary_set_assignments (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    set_id      uuid        NOT NULL REFERENCES vocabulary_sets(id) ON DELETE CASCADE,
    class_id    uuid        REFERENCES classes(id) ON DELETE CASCADE,
    student_id  uuid        REFERENCES profiles(id) ON DELETE CASCADE,
    assigned_by uuid        NOT NULL REFERENCES profiles(id),
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT vocabulary_set_assignment_target_check CHECK (
        (class_id IS NOT NULL AND student_id IS NULL) OR
        (class_id IS NULL AND student_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS vocabulary_set_assignments_set_id_idx
    ON vocabulary_set_assignments(set_id);
CREATE INDEX IF NOT EXISTS vocabulary_set_assignments_class_id_idx
    ON vocabulary_set_assignments(class_id);
CREATE INDEX IF NOT EXISTS vocabulary_set_assignments_student_id_idx
    ON vocabulary_set_assignments(student_id);
