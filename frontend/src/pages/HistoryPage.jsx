import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getHistory } from "../features/history/historyStorage";
import { saveLatestResult } from "../features/result/resultStorage";

export default function HistoryPage() {
    const [history, setHistory] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        setHistory(getHistory());
    }, []);

    function handleSelect(item) {
        saveLatestResult(item);
        navigate("/result");
    }

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8">
            <div className="mx-auto max-w-5xl">
                <h1 className="mb-6 text-3xl font-extrabold">
                    Practice History
                </h1>

                {history.length === 0 && (
                    <p className="text-slate-500">
                        No practice history yet.
                    </p>
                )}

                <div className="space-y-4">
                    {history.map((item) => (
                        <div
                            key={item.id}
                            onClick={() => handleSelect(item)}
                            className="cursor-pointer rounded-2xl bg-white p-5 shadow-sm transition hover:shadow-md"
                        >
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-slate-400">
                                        {new Date(item.analyzedAt).toLocaleString()}
                                    </p>

                                    <p className="text-xl font-extrabold">
                                        {item.word}
                                    </p>
                                </div>

                                <div
                                    className={`rounded-full px-4 py-2 font-extrabold ${item.score >= 85
                                        ? "bg-emerald-100 text-emerald-600"
                                        : item.score >= 70
                                            ? "bg-yellow-100 text-yellow-600"
                                            : "bg-red-100 text-red-600"
                                        }`}
                                >
                                    {item.score}
                                </div>
                            </div>

                            <div className="mt-3 flex gap-2">
                                {item.phonemes.map((p, i) => (
                                    <span
                                        key={i}
                                        className={`rounded-lg px-2 py-1 text-xs font-bold ${p.correct
                                            ? "bg-emerald-50 text-emerald-600"
                                            : "bg-red-50 text-red-600"
                                            }`}
                                    >
                                        {p.symbol}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </main>
    );
}