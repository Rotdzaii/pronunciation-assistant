import { useEffect, useRef, useState } from "react";
import {
    getAssignmentById,
    updateAssignmentItemResult,
} from "../assignmentStorage";
import {
    getPronunciationJobStatus,
    submitPronunciationAudio,
} from "../../../services/pronunciationApi";

export const useAssignmentPractice = (assignmentId) => {
    const [assignment, setAssignment] = useState(null);
    const [currentItem, setCurrentItem] = useState(null);

    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState("");

    const [status, setStatus] = useState("idle");
    const [jobId, setJobId] = useState(null);
    const [jobMessage, setJobMessage] = useState("");
    const [errorMessage, setErrorMessage] = useState("");

    const pollingRef = useRef(null);

    useEffect(() => {
        const data = getAssignmentById(assignmentId);

        setAssignment(data);
        setCurrentItem(data?.items?.find((item) => !item.completed) || null);
    }, [assignmentId]);

    useEffect(() => {
        return () => {
            if (audioUrl) URL.revokeObjectURL(audioUrl);
            if (pollingRef.current) clearTimeout(pollingRef.current);
        };
    }, [audioUrl]);

    useEffect(() => {
        if (!jobId) return;
        if (!currentItem) return;
        if (!["queued", "processing"].includes(status)) return;

        pollingRef.current = setTimeout(async () => {
            try {
                const res = await getPronunciationJobStatus(jobId);

                setJobMessage(res.message || "");

                if (res.status === "queued") {
                    setStatus("queued");
                    return;
                }

                if (res.status === "processing") {
                    setStatus("processing");
                    return;
                }

                if (res.status === "completed") {
                    updateAssignmentItemResult({
                        assignmentId,
                        itemId: currentItem.id,
                        score: res.result.score,
                        audioUrl,
                        phonemeErrors: res.result.phonemeErrors,
                    });

                    const updatedAssignment = getAssignmentById(assignmentId);
                    const nextItem =
                        updatedAssignment?.items?.find((item) => !item.completed) || null;

                    setAssignment(updatedAssignment);
                    setCurrentItem(nextItem);

                    setStatus("completed");
                    setAudioBlob(null);
                    setJobId(null);
                    return;
                }

                if (res.status === "failed") {
                    setStatus("failed");
                    setErrorMessage(res.message || "Assignment analysis failed.");
                }
            } catch (error) {
                console.error(error);
                setStatus("failed");
                setErrorMessage("Cannot connect to pronunciation API.");
            }
        }, 1500);

        return () => clearTimeout(pollingRef.current);
    }, [jobId, status, currentItem, assignmentId, audioUrl]);

    function handleAudio(blob) {
        if (audioUrl) URL.revokeObjectURL(audioUrl);

        const url = URL.createObjectURL(blob);

        setAudioBlob(blob);
        setAudioUrl(url);
        setStatus("recorded");
        setJobId(null);
        setJobMessage("");
        setErrorMessage("");
    }

    async function analyze() {
        if (!audioBlob || !currentItem) return;

        try {
            setStatus("submitting");
            setErrorMessage("");

            const res = await submitPronunciationAudio({
                audioBlob,
                transcript: currentItem.transcript || currentItem.text,
                assignmentId,
                itemId: currentItem.id,
            });

            setJobId(res.job_id);
            setStatus(res.status || "queued");
            setJobMessage("Assignment audio submitted. Waiting for AI analysis...");
        } catch (error) {
            console.error(error);
            setStatus("failed");
            setErrorMessage("Failed to submit assignment audio.");
        }
    }

    function resetCurrentAudio() {
        if (audioUrl) URL.revokeObjectURL(audioUrl);

        setAudioBlob(null);
        setAudioUrl("");
        setStatus("idle");
        setJobId(null);
        setJobMessage("");
        setErrorMessage("");
    }

    return {
        assignment,
        currentItem,
        audioUrl,
        status,
        jobMessage,
        errorMessage,
        handleAudio,
        analyze,
        resetCurrentAudio,
    };
};