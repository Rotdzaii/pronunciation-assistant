import { useNavigate } from "react-router-dom";
import PronunciationScore from "./PronunciationScore";
import PhonemeHighlighter from "./PhonemeHighlighter";
import SuggestionCard from "./SuggestionCard";

const mockResult = {
    word: "computer",
    score: 78,
    phonemes: [
        { symbol: "/k/", correct: true },
        { symbol: "/ə/", correct: true },
        { symbol: "/m/", correct: true },
        { symbol: "/pjuː/", correct: false },
        { symbol: "/tər/", correct: true },
    ],
    suggestion:
        "Try pronouncing /pjuː/ more clearly. Keep your lips rounded and release the /j/ sound before the long /uː/.",
};

export default function ResultPage() {
    const navigate = useNavigate();

    const wrongPhonemes = mockResult.phonemes.filter(
        (phoneme) => !phoneme.correct
    );

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8 text-slate-900">
            <div className="mx-auto max-w-6xl">
                <header className="mb-8 flex items-center justify-between">
                    <div>
                        <p className="text-sm font-extrabold uppercase text-purple-500">
                            AI Pronunciation Result
                        </p>

                        <h1 className="mt-2 text-4xl font-extrabold">
                            Kết quả phân tích phát âm
                        </h1>
                    </div>

                    <button
                        type="button"
                        onClick={() => navigate("/practice")}
                        className="rounded-2xl bg-purple-600 px-6 py-3 font-extrabold text-white shadow-sm"
                    >
                        Practice Again
                    </button>
                </header>

                <div className="grid grid-cols-[360px_1fr] gap-8">
                    <div className="space-y-6">
                        <PronunciationScore score={mockResult.score} />

                        <SuggestionCard suggestion={mockResult.suggestion} />
                    </div>

                    <div className="space-y-6">
                        <section className="rounded-3xl bg-white p-8 shadow-sm">
                            <p className="text-sm font-extrabold uppercase text-purple-500">
                                Target Word
                            </p>

                            <h2 className="mt-4 text-6xl font-extrabold text-slate-900">
                                {mockResult.word}
                            </h2>

                            <p className="mt-4 text-slate-500">
                                AI đã phân tích phát âm của bạn theo từng âm vị.
                            </p>
                        </section>

                        <PhonemeHighlighter phonemes={mockResult.phonemes} />

                        {wrongPhonemes.length > 0 && (
                            <section className="rounded-3xl bg-red-50 p-8">
                                <p className="text-sm font-extrabold uppercase text-red-600">
                                    Need Practice
                                </p>

                                <h3 className="mt-3 text-2xl font-extrabold text-red-700">
                                    Bạn cần luyện thêm:
                                </h3>

                                <div className="mt-5 flex flex-wrap gap-3">
                                    {wrongPhonemes.map((phoneme) => (
                                        <span
                                            key={phoneme.symbol}
                                            className="rounded-2xl bg-white px-5 py-3 text-lg font-extrabold text-red-600"
                                        >
                                            {phoneme.symbol}
                                        </span>
                                    ))}
                                </div>
                            </section>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}