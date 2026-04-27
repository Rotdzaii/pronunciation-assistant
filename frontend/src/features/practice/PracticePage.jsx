import { useEffect, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import AudioUploader from "./AudioUploader";

export default function PracticePage() {
    const [audioBlob, setAudioBlob] = useState(null);
    const [audioUrl, setAudioUrl] = useState("");

    const sentence = "The quick brown fox jumps over the lazy dog.";
    const hasAudio = Boolean(audioBlob);

    useEffect(() => {
        return () => {
            if (audioUrl) URL.revokeObjectURL(audioUrl);
        };
    }, [audioUrl]);

    function handleAudioReady(blob) {
        if (audioUrl) URL.revokeObjectURL(audioUrl);

        const url = URL.createObjectURL(blob);

        setAudioBlob(blob);
        setAudioUrl(url);
    }

    function resetAudio() {
        if (audioUrl) URL.revokeObjectURL(audioUrl);

        setAudioBlob(null);
        setAudioUrl("");
    }

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
                    <AudioRecorder onAudioReady={handleAudioReady} disabled={false} />
                </div>

                <div className="mt-12 grid grid-cols-3 gap-5">
                    <button
                        type="button"
                        disabled={!audioUrl}
                        onClick={() => {
                            const audio = new Audio(audioUrl);
                            audio.play();
                        }}
                        className="rounded-3xl bg-white px-8 py-5 font-bold shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        ▶ Replay
                    </button>

                    <AudioUploader onAudioReady={handleAudioReady} disabled={false} />

                    <button
                        type="button"
                        disabled={!audioUrl}
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

                <button
                    type="button"
                    disabled={!hasAudio}
                    className="mt-8 rounded-2xl bg-purple-600 px-10 py-4 text-lg font-extrabold text-white shadow-lg transition hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                    Analyze Pronunciation
                </button>

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