import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import AppLayout from "../layouts/AppLayout";
import AudioRecorder from "../features/practice/AudioRecorder";
import AudioUploader from "../features/practice/AudioUploader";
import { saveLatestResult } from "../features/result/resultStorage";
import { addToHistory } from "../features/history/historyStorage";

async function mockSubmitAudio() {
    await new Promise((r) => setTimeout(r, 1000));
    return { job_id: "job_mock_001", status: "queued" };
}

async function mockGetJobStatus(jobId, pollCount) {
    await new Promise((r) => setTimeout(r, 700));

    if (pollCount <= 1) return { status: "queued", message: "Waiting..." };
    if (pollCount <= 3) return { status: "processing", message: "Analyzing..." };

    return {
        status: "completed",
        result_id: "result_mock_001",
        result: {
            score: 78,
            word: "computer",
            phonemes: [
                { symbol: "/k/", correct: true },
                { symbol: "/ə/", correct: true },
                { symbol: "/m/", correct: true },
                { symbol: "/pjuː/", correct: false },
                { symbol: "/tər/", correct: true },
            ],
            suggestion: "Improve /pjuː/ pronunciation",
        },
    };
}

export default function PracticePage() {
    const navigate = useNavigate();
    const pollingRef = useRef(null);

    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState("");
    const [status, setStatus] = useState("idle");
    const [jobId, setJobId] = useState(null);
    const [pollCount, setPollCount] = useState(0);
    const [result, setResult] = useState(null);

    const isBusy = ["submitting", "queued", "processing"].includes(status);

    useEffect(() => {
        if (!jobId) return;

        pollingRef.current = setTimeout(async () => {
            const next = pollCount + 1;
            const res = await mockGetJobStatus(jobId, next);

            setPollCount(next);

            if (res.status === "completed") {
                const finalResult = {
                    ...res.result,
                    id: res.result_id,
                    audioUrl,
                    analyzedAt: new Date().toISOString(),
                };

                saveLatestResult(finalResult);
                addToHistory(finalResult);
                setResult(finalResult);
                setStatus("completed");
            } else {
                setStatus(res.status);
            }
        }, 1500);

        return () => clearTimeout(pollingRef.current);
    }, [jobId, pollCount]);

    function handleAudio(blob) {
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);
        setStatus("recorded");
    }

    async function analyze() {
        if (!audioBlob) return;

        setStatus("submitting");
        const res = await mockSubmitAudio();
        setJobId(res.job_id);
        setStatus("queued");
    }

    return (
        <AppLayout>
            <h1 className="text-3xl font-bold">Practice</h1>
            <AudioRecorder onAudioReady={handleAudio} disabled={isBusy} />
            <AudioUploader onAudioReady={handleAudio} disabled={isBusy} />

            <button onClick={analyze}>Analyze</button>

            {result && (
                <button onClick={() => navigate("/result")}>
                    View Result
                </button>
            )}
        </AppLayout>
    );
}