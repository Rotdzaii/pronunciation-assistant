import { useNavigate, useParams } from "react-router-dom";
import AudioRecorder from "../features/practice/AudioRecorder";
import AudioUploader from "../features/practice/AudioUploader";
import { useAssignmentPractice } from "../features/assignments/hooks/useAssignmentPractice";

export default function AssignmentPracticePage() {
    const { assignmentId } = useParams();
    const navigate = useNavigate();

    const {
        assignment,
        currentItem,
        audioUrl,
        status,
        jobMessage,
        errorMessage,
        handleAudio,
        analyze,
        resetCurrentAudio,
    } = useAssignmentPractice(assignmentId);

    const isBusy = ["submitting", "queued", "processing"].includes(status);

    if (!assignment) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-[#f7f1ff] p-8">
                <div className="rounded-3xl bg-white p-8 text-center shadow-sm">
                    <p className="text-lg font-extrabold text-purple-600">
                        Loading assignment practice...
                    </p>
                </div>
            </main>
        );
    }

    if (!currentItem) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-[#f7f1ff] p-8">
                <div className="max-w-xl rounded-3xl bg-white p-10 text-center shadow-sm">
                    <div className="mx-auto mb-5 flex h-20 w-20 items-center justify-center rounded-full bg-green-50 text-4xl">
                        🎉
                    </div>

                    <h1 className="text-3xl font-extrabold text-slate-900">
                        Assignment Completed!
                    </h1>

                    <p className="mt-3 text-slate-500">
                        Bạn đã luyện xong tất cả item trong assignment này.
                    </p>

                    <button
                        onClick={() => navigate(`/assignments/${assignmentId}`)}
                        className="mt-6 rounded-2xl bg-purple-600 px-6 py-3 font-extrabold text-white"
                    >
                        Back to Detail
                    </button>
                </div>
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8 text-slate-900">
            <div className="mx-auto max-w-5xl space-y-6">
                <button
                    type="button"
                    onClick={() => navigate(`/assignments/${assignmentId}`)}
                    className="text-sm font-extrabold text-purple-600"
                >
                    ← Back to Assignment Detail
                </button>

                <header className="rounded-3xl bg-purple-600 p-6 text-white shadow-sm">
                    <p className="text-sm font-extrabold uppercase text-purple-100">
                        Assignment · Real API Mode
                    </p>

                    <h1 className="mt-2 text-3xl font-extrabold">{assignment.title}</h1>

                    <p className="mt-2 text-purple-100">
                        Submit audio to backend, receive job_id, then poll real AI result.
                    </p>
                </header>

                <section className="rounded-3xl bg-white p-8 text-center shadow-sm">
                    <p className="text-sm font-extrabold uppercase text-purple-500">
                        Current Practice Item
                    </p>

                    <h2 className="mt-3 text-4xl font-extrabold text-slate-900">
                        {currentItem.text}
                    </h2>

                    {currentItem.note && (
                        <p className="mx-auto mt-3 max-w-2xl text-slate-500">
                            {currentItem.note}
                        </p>
                    )}

                    <div className="mt-5 flex justify-center gap-3">
                        <span className="rounded-full bg-purple-50 px-4 py-2 text-sm font-bold text-purple-600">
                            Target: {currentItem.targetPhoneme || "Sentence focus"}
                        </span>

                        <span className="rounded-full bg-slate-100 px-4 py-2 text-sm font-bold text-slate-600">
                            Previous score: {currentItem.previousScore || "--"}
                        </span>
                    </div>

                    <div className="mt-10">
                        <AudioRecorder onAudioReady={handleAudio} disabled={isBusy} />
                    </div>

                    <div className="mt-8 flex justify-center gap-4">
                        <button
                            type="button"
                            disabled={!audioUrl}
                            onClick={() => new Audio(audioUrl).play()}
                            className="rounded-2xl bg-slate-100 px-5 py-3 font-extrabold text-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Replay
                        </button>

                        <AudioUploader onAudioReady={handleAudio} disabled={isBusy} />

                        <button
                            type="button"
                            onClick={resetCurrentAudio}
                            disabled={isBusy}
                            className="rounded-2xl bg-slate-100 px-5 py-3 font-extrabold text-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Reset
                        </button>
                    </div>

                    {audioUrl && <audio controls src={audioUrl} className="mt-6 w-full" />}

                    <button
                        type="button"
                        onClick={analyze}
                        disabled={!audioUrl || isBusy}
                        className="mt-6 rounded-2xl bg-purple-600 px-8 py-4 font-extrabold text-white disabled:cursor-not-allowed disabled:bg-slate-300"
                    >
                        {status === "submitting"
                            ? "Submitting..."
                            : status === "queued"
                                ? "Waiting..."
                                : status === "processing"
                                    ? "Analyzing..."
                                    : "Submit Assignment Audio"}
                    </button>

                    {jobMessage && (
                        <p className="mx-auto mt-5 max-w-xl rounded-2xl bg-purple-50 px-5 py-3 text-sm font-bold text-purple-600">
                            {jobMessage}
                        </p>
                    )}

                    {errorMessage && (
                        <p className="mx-auto mt-5 max-w-xl rounded-2xl bg-red-50 px-5 py-3 text-sm font-bold text-red-600">
                            {errorMessage}
                        </p>
                    )}

                    {status === "completed" && (
                        <p className="mx-auto mt-5 max-w-xl rounded-2xl bg-green-50 px-5 py-3 text-sm font-bold text-green-600">
                            Result saved. Moving to next item if available.
                        </p>
                    )}
                </section>
            </div>
        </main>
    );
}