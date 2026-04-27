import { useEffect, useMemo, useState } from "react";
import { getAssignments } from "../assignmentStorage";
import { ASSIGNMENT_STATUS } from "../mockAssignments";

export const useTeacherAssignments = () => {
    const [assignments, setAssignments] = useState([]);

    useEffect(() => {
        setAssignments(getAssignments());
    }, []);

    const stats = useMemo(() => {
        const total = assignments.length;

        const completed = assignments.filter(
            (assignment) => assignment.status === ASSIGNMENT_STATUS.COMPLETED
        ).length;

        const inProgress = assignments.filter(
            (assignment) => assignment.status === ASSIGNMENT_STATUS.IN_PROGRESS
        ).length;

        const late = assignments.filter(
            (assignment) => assignment.status === ASSIGNMENT_STATUS.LATE
        ).length;

        const newCount = assignments.filter(
            (assignment) => assignment.status === ASSIGNMENT_STATUS.NEW
        ).length;

        const averageProgress =
            total > 0
                ? Math.round(
                    assignments.reduce(
                        (sum, assignment) => sum + Number(assignment.progress?.percent || 0),
                        0
                    ) / total
                )
                : 0;

        return {
            total,
            completed,
            inProgress,
            late,
            newCount,
            averageProgress,
        };
    }, [assignments]);

    return {
        assignments,
        stats,
    };
};