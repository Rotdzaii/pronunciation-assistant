import { useEffect, useRef, useState } from "react";
import {
    getAssignmentById,
    updateAssignmentItemResult,
} from "../assignmentStorage";

async function mockSubmitAudio() {
    await new Promise((r) => setTimeout(r, 800));
    return { job_id: "job_assignment_001" };
}

async function mockPolling(jobId, pollCount) {
    await new Promise((r) => setTimeout(r, 600));

    if (pollCount < 2) return { status: "processing" };

    return {
        status: "completed",
        score: Math.floor(60 + Math.random() * 40),
        phonemeErrors: [],
    };
}

export const useAssignmentPractice = (assignmentId) => {
    const [assignment, setAssignment] = useState(null);
    const [currentItem, setCurrentItem] = useState(null);

    const [audioBlob, setAudioBlob] = useState(null);
    const [status, setStatus] = useState("idle");
    const [jobId, setJobId] = useState(null);
    const [pollCount, setPollCount] = useState(0);

    const pollingRef = useRef(null);

    useEffect(() => {
        const data = getAssignmentById(assignmentId);
        setAssignment(data);
        setCurrentItem(data?.items?.find((i) => !i.completed));
    }, [assignmentId]);

    useEffect(() => {
        if (!jobId) return;

        pollingRef.current = setTimeout(async () => {
            const next = pollCount + 1;
            const res = await mockPolling(jobId, next);

            setPollCount(next);

            if (res.status === "completed") {
                updateAssignmentItemResult({
                    assignmentId,
                    itemId: currentItem.id,
                    score: res.score,
                    phonemeErrors: res.phonemeErrors,
                });

                const updated = getAssignmentById(assignmentId);
                setAssignment(updated);

                const nextItem = updated.items.find((i) => !i.completed);
                setCurrentItem(nextItem || null);

                setStatus("completed");
            } else {
                setStatus("processing");
            }
        }, 1200);

        return () => clearTimeout(pollingRef.current);
    }, [jobId, pollCount]);

    function handleAudio(blob) {
        setAudioBlob(blob);
        setStatus("recorded");
    }

    async function analyze() {
        if (!audioBlob || !currentItem) return;

        setStatus("submitting");
        const res = await mockSubmitAudio();

        setJobId(res.job_id);
        setStatus("processing");
    }

    return {
        assignment,
        currentItem,
        status,
        handleAudio,
        analyze,
    };
};