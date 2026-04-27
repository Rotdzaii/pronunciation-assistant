import { useEffect, useMemo, useState } from "react";
import { getAssignmentById, updateAssignment } from "../assignmentStorage";

export const useTeacherReview = (assignmentId) => {
    const [assignment, setAssignment] = useState(null);
    const [comment, setComment] = useState("");

    useEffect(() => {
        const data = getAssignmentById(assignmentId);

        if (data) {
            setAssignment(data);
            setComment(data.teacherFeedback || "");
        }
    }, [assignmentId]);

    const improvement = useMemo(() => {
        if (!assignment) return null;

        const items = assignment.items || [];

        const improvements = items
            .filter(
                (item) =>
                    typeof item.previousScore === "number" &&
                    typeof item.latestScore === "number"
            )
            .map((item) => ({
                text: item.text,
                before: item.previousScore,
                after: item.latestScore,
                diff: item.latestScore - item.previousScore,
            }));

        return improvements;
    }, [assignment]);

    const saveComment = () => {
        if (!assignment) return;

        const updated = updateAssignment(assignment.id, {
            teacherFeedback: comment,
        });

        setAssignment(updated);
    };

    return {
        assignment,
        comment,
        setComment,
        improvement,
        saveComment,
    };
};