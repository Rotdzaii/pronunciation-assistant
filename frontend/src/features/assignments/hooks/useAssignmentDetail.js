import { useEffect, useState } from "react";
import { getAssignmentById, markAssignmentAsViewed } from "../assignmentStorage";

export const useAssignmentDetail = (assignmentId) => {
    const [assignment, setAssignment] = useState(null);

    useEffect(() => {
        if (!assignmentId) return;

        const data = getAssignmentById(assignmentId);

        if (data) {
            markAssignmentAsViewed(assignmentId);
            setAssignment(data);
        }
    }, [assignmentId]);

    return {
        assignment,
    };
};