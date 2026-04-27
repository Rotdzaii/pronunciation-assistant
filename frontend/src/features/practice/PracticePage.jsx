import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import AppLayout from "../../layouts/AppLayout";
import AudioRecorder from "./AudioRecorder";
import AudioUploader from "./AudioUploader";
import { saveLatestResult } from "../result/resultStorage";
import { addToHistory } from "../history/historyStorage";

async function mockSubmitAudio() {
    await new Promise((r) => setTimeout(r, 1000));
    return { job_id: "job_mock_001", status: "queued" };
}

async function mockGetJobStatus(jobId, pollCount) {
    await new Promise((r) => setTimeout(r, 700));

    if (pollCount <= 1) {
        return { status: "queued", message: "Waiting in queue..." };
    }

    if (pollCount <= 3) {
        return { status: "processing", message: "Analyzing phonemes..." };
    }

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
            suggestion:
                "Try pronouncing /pjuː/ more clearly. Keep lips rounded.",
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
    const [jobMessage, setJobMessage] = useState("");
    const [result, setResult] = useState(null);

    const sentence = "The quick brown fox jumps over the lazy dog.";

    const isBusy = ["submitting", "queued", "processing"].includes(status);


    useEffect(() => {
        return () => {
            if (audioUrl) URL.revokeObjectURL(audioUrl);
            if (pollingRef.current) clearTimeout(pollingRef.current);
        };
    }, [audioUrl]);


    useEffect(() => {
        if (!jobId) return;
        if (!["queued", "processing"].includes(status)) return;

        pollingRef.current = setTimeout(async () => {
            const next = pollCount + 1;
            const res = await mockGetJobStatus(jobId, next);

            setPollCount(next);
            setJobMessage(res.message || "");

            if (res.status === "queued") return setStatus("queued");

            if (res.status === "processing") return setStatus("processing");

            if (res.status === "completed") {
                const finalResult = {
                    ...res.result,
                    id: res.result_id,
                    audioUrl,
                    sentence,
                    analyzedAt: new Date().toISOString(),
                };

                saveLatestResult(finalResult);
                addToHistory(finalResult);

                setResult(finalResult);
                setStatus("completed");
            }
        }, 1500);

        return () => clearTimeout(pollingRef.current);
    }, [jobId, status, pollCount, audioUrl, sentence]);

    function handleAudio(blob) {
        if (audioUrl) URL.revokeObjectURL(audioUrl);

        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);

        setStatus("recorded");
        setResult(null);
        setJobId(null);
    }

    function reset() {
        if (isBusy) return;

        if (audioUrl) URL.revokeObjectURL(audioUrl);

        setAudioBlob(null);
        setAudioUrl("");
        setStatus("idle");
        setResult(null);
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
            <section className="mx-auto flex min-h-[700px] max-w-5xl flex-col items-center justify-center rounded-3xl bg-white p-10 shadow-sm">
                <h1 className="text-center text-4xl font-extrabold">
                    "{sentence}"
                </h1>

                <div className="mt-16">
                    <AudioRecorder onAudioReady={handleAudio} disabled={isBusy} />
                </div>
                <div className="mt-10 flex gap-4">
                    <button
                        disabled={!audioUrl}
                        onClick={() => new Audio(audioUrl).play()}
                        className="btn"
                    >
                        Replay
                    </button>

                    <AudioUploader onAudioReady={handleAudio} disabled={isBusy} />

                    <button onClick={reset} className="btn">
                        Reset
                    </button>
                </div>

                {audioUrl && (
                    <audio controls src={audioUrl} className="mt-6 w-full" />
                )}
                <button
                    onClick={analyze}
                    disabled={!audioBlob || isBusy}
                    className="mt-6 rounded-xl bg-purple-600 px-6 py-3 font-bold text-white"
                >
                    {status === "processing"
                        ? "Analyzing..."
                        : status === "queued"
                            ? "Waiting..."
                            : "Analyze Pronunciation"}
                </button>

                {status !== "idle" && (
                    <div className="mt-6 text-sm text-purple-600">
                        {jobMessage}
                    </div>
                )}

                {result && (
                    <div className="mt-8 w-full rounded-2xl bg-emerald-50 p-5">
                        <p className="font-bold text-emerald-700">
                            Result ready ({result.score}/100)
                        </p>

                        <div className="mt-3 flex gap-2">
                            {result.phonemes.map((p) => (
                                <span
                                    key={p.symbol}
                                    className={`px-3 py-1 rounded ${p.correct
                                        ? "bg-green-200"
                                        : "bg-red-200 text-red-700"
                                        }`}
                                >
                                    {p.symbol}
                                </span>
                            ))}
                        </div>

                        <button
                            onClick={() => navigate("/result")}
                            className="mt-4 w-full rounded-xl bg-green-600 py-2 text-white font-bold"
                        >
                            View Full Result
                        </button>
                    </div>
                )}
            </section>
        </AppLayout>
    );
}