import { useEffect, useState } from "react";
import { getAssignments } from "../assignmentStorage";

export const useAssignments = () => {
    const [assignments, setAssignments] = useState([]);

    useEffect(() => {
        const data = getAssignments();
        setAssignments(data);
    }, []);

    return {
        assignments,
    };
};