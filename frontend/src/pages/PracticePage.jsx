import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import AppLayout from "../layouts/AppLayout";
import AudioRecorder from "../features/practice/AudioRecorder";
import AudioUploader from "../features/practice/AudioUploader";
import { saveLatestResult } from "../features/result/resultStorage";
import { addToHistory } from "../features/history/historyStorage";
import {
    getPronunciationJobStatus,
    submitPronunciationAudio,
} from "../services/pronunciationApi";

export default function PracticePage() {
    const navigate = useNavigate();
    const pollingRef = useRef(null);

    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState("");

    const [status, setStatus] = useState("idle");
    const [jobId, setJobId] = useState(null);
    const [jobMessage, setJobMessage] = useState("");
    const [result, setResult] = useState(null);
    const [errorMessage, setErrorMessage] = useState("");

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
                    const finalResult = {
                        ...res.result,
                        id: res.result_id || jobId,
                        audioUrl,
                        sentence,
                        analyzedAt: new Date().toISOString(),
                    };

                    saveLatestResult(finalResult);
                    addToHistory(finalResult);

                    setResult(finalResult);
                    setStatus("completed");
                    return;
                }

                if (res.status === "failed") {
                    setStatus("failed");
                    setErrorMessage(res.message || "AI analysis failed.");
                }
            } catch (error) {
                console.error(error);
                setStatus("failed");
                setErrorMessage("Cannot connect to pronunciation API.");
            }
        }, 1500);

        return () => clearTimeout(pollingRef.current);
    }, [jobId, status, audioUrl, sentence]);

    function handleAudio(blob) {
        if (audioUrl) URL.revokeObjectURL(audioUrl);

        const url = URL.createObjectURL(blob);

        setAudioBlob(blob);
        setAudioUrl(url);
        setStatus("recorded");
        setResult(null);
        setJobId(null);
        setJobMessage("");
        setErrorMessage("");
    }

    function reset() {
        if (isBusy) return;

        if (audioUrl) URL.revokeObjectURL(audioUrl);

        setAudioBlob(null);
        setAudioUrl("");
        setStatus("idle");
        setResult(null);
        setJobId(null);
        setJobMessage("");
        setErrorMessage("");
    }

    async function analyze() {
        if (!audioBlob) return;

        try {
            setStatus("submitting");
            setErrorMessage("");

            const res = await submitPronunciationAudio({
                audioBlob,
                transcript: sentence,
            });

            setJobId(res.job_id);
            setStatus(res.status || "queued");
            setJobMessage("Audio submitted. Waiting for AI analysis...");
        } catch (error) {
            console.error(error);
            setStatus("failed");
            setErrorMessage("Failed to submit audio. Please check backend API.");
        }
    }

    return (
        <AppLayout>
            <section className="mx-auto flex min-h-[700px] max-w-5xl flex-col items-center justify-center rounded-3xl bg-white p-10 shadow-sm">
                <p className="mb-3 rounded-full bg-purple-50 px-4 py-2 text-sm font-extrabold text-purple-600">
                    Real API Mode
                </p>

                <h1 className="text-center text-4xl font-extrabold">"{sentence}"</h1>

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

                {audioUrl && <audio controls src={audioUrl} className="mt-6 w-full" />}

                <button
                    onClick={analyze}
                    disabled={!audioBlob || isBusy}
                    className="mt-6 rounded-xl bg-purple-600 px-6 py-3 font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                    {status === "submitting"
                        ? "Submitting..."
                        : status === "processing"
                            ? "Analyzing..."
                            : status === "queued"
                                ? "Waiting..."
                                : "Analyze Pronunciation"}
                </button>

                {jobMessage && (
                    <div className="mt-6 rounded-2xl bg-purple-50 px-5 py-3 text-sm font-bold text-purple-600">
                        {jobMessage}
                    </div>
                )}

                {errorMessage && (
                    <div className="mt-6 rounded-2xl bg-red-50 px-5 py-3 text-sm font-bold text-red-600">
                        {errorMessage}
                    </div>
                )}

                {result && (
                    <div className="mt-8 w-full rounded-2xl bg-emerald-50 p-5">
                        <p className="font-bold text-emerald-700">
                            Result ready ({result.score}/100)
                        </p>

                        <div className="mt-3 flex flex-wrap gap-2">
                            {result.phonemes.map((p, index) => (
                                <span
                                    key={`${p.symbol}-${index}`}
                                    className={`rounded px-3 py-1 ${p.correct ? "bg-green-200" : "bg-red-200 text-red-700"
                                        }`}
                                >
                                    {p.symbol}
                                </span>
                            ))}
                        </div>

                        <button
                            onClick={() => navigate("/result")}
                            className="mt-4 w-full rounded-xl bg-green-600 py-2 font-bold text-white"
                        >
                            View Full Result
                        </button>
                    </div>
                )}
            </section>
        </AppLayout>
    );
}