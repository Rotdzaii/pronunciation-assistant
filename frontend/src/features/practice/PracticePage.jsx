import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import AudioRecorder from "./AudioRecorder";
import AudioUploader from "./AudioUploader";

async function mockSubmitAudio() {
    await new Promise((resolve) => setTimeout(resolve, 1000));

    return {
        job_id: "job_mock_001",
        status: "queued",
    };
}

async function mockGetJobStatus(jobId, pollCount) {
    await new Promise((resolve) => setTimeout(resolve, 700));

    if (!jobId) {
        return {
            status: "failed",
            message: "Missing job_id",
        };
    }

    if (pollCount <= 1) {
        return {
            status: "queued",
            message: "Audio is waiting in queue",
        };
    }

    if (pollCount <= 3) {
        return {
            status: "processing",
            message: "AI is mapping audio to phonemes",
        };
    }

    return {
        status: "completed",
        result_id: "result_mock_001",
        message: "Pronunciation analysis completed",
        result: {
            score: 78,
            word: "computer",
            phonemes: [
                { symbol: "/k/", correct: true },
                { symbol: "/pjuː/", correct: false },
            ],
            suggestion: "Try pronouncing /pjuː/ more clearly.",
        },
    };
}

export default function PracticePage() {
    const navigate = useNavigate();
    const pollingTimerRef = useRef(null);

    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState("");

    const [status, setStatus] = useState("idle");
    const [jobId, setJobId] = useState(null);
    const [pollCount, setPollCount] = useState(0);
    const [jobMessage, setJobMessage] = useState("");
    const [mockResult, setMockResult] = useState(null);
    const [error, setError] = useState("");

    const sentence = "The quick brown fox jumps over the lazy dog.";

    const hasAudio = Boolean(audioBlob);
    const isSubmitting = status === "submitting";
    const isQueued = status === "queued";
    const isProcessing = status === "processing";
    const isCompleted = status === "completed";
    const isBusy = isSubmitting || isQueued || isProcessing;

    useEffect(() => {
        return () => {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }

            if (pollingTimerRef.current) {
                clearTimeout(pollingTimerRef.current);
            }
        };
    }, [audioUrl]);

    useEffect(() => {
        if (!jobId) return;
        if (status !== "queued" && status !== "processing") return;

        pollingTimerRef.current = setTimeout(async () => {
            try {
                const nextPollCount = pollCount + 1;
                const response = await mockGetJobStatus(jobId, nextPollCount);

                setPollCount(nextPollCount);
                setJobMessage(response.message || "");

                if (response.status === "queued") {
                    setStatus("queued");
                    return;
                }

                if (response.status === "processing") {
                    setStatus("processing");
                    return;
                }

                if (response.status === "completed") {
                    setStatus("completed");
                    setMockResult(response.result);
                    return;
                }

                if (response.status === "failed") {
                    setStatus("failed");
                    setError(response.message || "AI analysis failed.");
                }
            } catch {
                setStatus("failed");
                setError("Mất kết nối khi đang polling trạng thái AI.");
            }
        }, 1500);

        return () => {
            if (pollingTimerRef.current) {
                clearTimeout(pollingTimerRef.current);
            }
        };
    }, [jobId, status, pollCount]);

    function handleAudioReady(blob) {
        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
        }

        const url = URL.createObjectURL(blob);

        setAudioBlob(blob);
        setAudioUrl(url);
        setStatus("recorded");
        setJobId(null);
        setPollCount(0);
        setJobMessage("");
        setMockResult(null);
        setError("");
    }

    function resetAudio() {
        if (isBusy) return;

        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
        }

        setAudioBlob(null);
        setAudioUrl("");
        setStatus("idle");
        setJobId(null);
        setPollCount(0);
        setJobMessage("");
        setMockResult(null);
        setError("");
    }

    async function handleAnalyze() {
        if (!audioBlob || isBusy) return;

        try {
            setStatus("submitting");
            setJobId(null);
            setPollCount(0);
            setJobMessage("Uploading audio...");
            setMockResult(null);
            setError("");

            const response = await mockSubmitAudio();

            setJobId(response.job_id);
            setStatus("queued");
            setJobMessage("Audio uploaded. Waiting for AI worker...");
        } catch {
            setStatus("failed");
            setError("Không thể gửi audio. Vui lòng thử lại.");
        }
    }

    function getProcessingStep() {
        if (isSubmitting) return 1;
        if (isQueued) return 2;
        if (isProcessing) return 3;
        if (isCompleted) return 4;
        return 0;
    }

    const processingStep = getProcessingStep();

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8 text-slate-900">
            <section className="mx-auto flex min-h-[700px] max-w-5xl flex-col items-center justify-center rounded-3xl bg-white p-10 shadow-sm">
                <div className="mb-3 rounded-full bg-blue-100 px-4 py-1 text-xs font-extrabold text-blue-600">
                    MASTER THIS PHRASE
                </div>

                <h1 className="max-w-3xl text-center text-5xl font-bold leading-tight">
                    "{sentence}"
                </h1>

                <p className="mt-3 rounded-full bg-slate-100 px-4 py-1 text-sm text-slate-500">
                    /ðə kwɪk braʊn fɑks dʒʌmps ˈoʊvər ðə ˈleɪzi dɔɡ/
                </p>

                <div className="mt-20">
                    <AudioRecorder onAudioReady={handleAudioReady} disabled={isBusy} />
                </div>

                <div className="mt-12 grid grid-cols-3 gap-5">
                    <button
                        type="button"
                        disabled={!audioUrl || isBusy}
                        onClick={() => {
                            const audio = new Audio(audioUrl);
                            audio.play();
                        }}
                        className="rounded-3xl bg-white px-8 py-5 font-bold shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        ▶ Replay
                    </button>

                    <AudioUploader onAudioReady={handleAudioReady} disabled={isBusy} />

                    <button
                        type="button"
                        disabled={!audioUrl || isBusy}
                        onClick={resetAudio}
                        className="rounded-3xl bg-white px-8 py-5 font-bold shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        🗑 Reset
                    </button>
                </div>

                {audioUrl && (
                    <div className="mt-8 w-full max-w-xl rounded-2xl bg-purple-50 p-4">
                        <p className="mb-2 text-sm font-bold text-purple-700">
                            Your recording
                        </p>
                        <audio controls src={audioUrl} className="w-full" />
                    </div>
                )}

                {error && (
                    <div className="mt-6 w-full max-w-xl rounded-2xl bg-red-50 p-4 text-sm font-bold text-red-600">
                        {error}
                    </div>
                )}

                <button
                    type="button"
                    disabled={!hasAudio || isBusy}
                    onClick={handleAnalyze}
                    className="mt-8 rounded-2xl bg-purple-600 px-10 py-4 text-lg font-extrabold text-white shadow-lg transition hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {isSubmitting
                        ? "Uploading audio..."
                        : isQueued
                            ? "Waiting for AI..."
                            : isProcessing
                                ? "AI is analyzing..."
                                : isCompleted
                                    ? "Analysis Completed"
                                    : "Analyze Pronunciation"}
                </button>

                {(isSubmitting || isQueued || isProcessing || isCompleted) && (
                    <div className="mt-6 w-full max-w-xl rounded-2xl bg-purple-50 p-5">
                        <p className="font-extrabold text-purple-700">
                            {isCompleted
                                ? "AI phân tích hoàn tất"
                                : "AI đang phân tích phát âm..."}
                        </p>

                        <p className="mt-1 text-sm text-purple-500">
                            {jobMessage || "Đang xử lý audio và mapping phoneme."}
                        </p>

                        <div className="mt-5 space-y-3">
                            <ProcessingStep
                                step={1}
                                currentStep={processingStep}
                                label="Upload audio"
                            />
                            <ProcessingStep
                                step={2}
                                currentStep={processingStep}
                                label="Queue AI job"
                            />
                            <ProcessingStep
                                step={3}
                                currentStep={processingStep}
                                label="Map speech to phonemes"
                            />
                            <ProcessingStep
                                step={4}
                                currentStep={processingStep}
                                label="Generate pronunciation feedback"
                            />
                        </div>

                        <div className="mt-4 h-2 overflow-hidden rounded-full bg-purple-100">
                            <div
                                className="h-full rounded-full bg-purple-600 transition-all"
                                style={{ width: `${processingStep * 25}%` }}
                            />
                        </div>

                        {jobId && (
                            <p className="mt-3 text-xs font-medium text-purple-400">
                                Job ID: {jobId}
                            </p>
                        )}
                    </div>
                )}

                {mockResult && (
                    <div className="mt-6 w-full max-w-xl rounded-2xl bg-emerald-50 p-5">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="font-extrabold text-emerald-700">
                                    Mock result ready
                                </p>
                                <p className="text-sm text-emerald-600">
                                    Bấm để xem trang kết quả chi tiết.
                                </p>
                            </div>

                            <div className="rounded-full bg-white px-4 py-2 font-extrabold text-emerald-600">
                                {mockResult.score}/100
                            </div>
                        </div>

                        <div className="mt-4 flex gap-3">
                            {mockResult.phonemes.map((phoneme) => (
                                <span
                                    key={phoneme.symbol}
                                    className={`rounded-2xl px-4 py-2 text-sm font-extrabold ${phoneme.correct
                                            ? "bg-emerald-100 text-emerald-700"
                                            : "bg-red-100 text-red-700"
                                        }`}
                                >
                                    {phoneme.correct ? "✓" : "!"} {phoneme.symbol}
                                </span>
                            ))}
                        </div>

                        <p className="mt-4 rounded-2xl bg-white p-4 text-sm font-medium text-slate-600">
                            💡 {mockResult.suggestion}
                        </p>

                        <button
                            type="button"
                            onClick={() => navigate("/result")}
                            className="mt-5 w-full rounded-2xl bg-emerald-600 px-6 py-3 font-extrabold text-white transition hover:bg-emerald-700"
                        >
                            View Full Result
                        </button>
                    </div>
                )}

                <div className="mt-8 max-w-xl rounded-2xl border border-dashed border-purple-200 bg-white p-5">
                    <p className="font-bold text-slate-800">✨ Teacher Tip</p>
                    <p className="text-sm text-slate-500">
                        Speak clearly and focus on final consonants like /ks/ in "fox".
                    </p>
                </div>
            </section>
        </main>
    );
}

function ProcessingStep({ step, currentStep, label }) {
    const isDone = currentStep > step;
    const isActive = currentStep === step;
    const isPending = currentStep < step;

    return (
        <div className="flex items-center gap-3">
            <div
                className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-extrabold ${isDone
                        ? "bg-emerald-500 text-white"
                        : isActive
                            ? "bg-purple-600 text-white"
                            : "bg-purple-100 text-purple-300"
                    }`}
            >
                {isDone ? "✓" : step}
            </div>

            <p
                className={`text-sm font-bold ${isPending ? "text-purple-300" : "text-purple-700"
                    }`}
            >
                {label}
            </p>
        </div>
    );
}