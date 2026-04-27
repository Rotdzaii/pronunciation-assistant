import { useEffect, useMemo, useState } from "react";
import { getHistory } from "../history/historyStorage";

function getScoreColor(score) {
    if (score >= 85) return "text-emerald-600 bg-emerald-50";
    if (score >= 70) return "text-yellow-600 bg-yellow-50";
    return "text-red-600 bg-red-50";
}

function getAverageScore(history) {
    if (history.length === 0) return 0;

    const total = history.reduce((sum, item) => sum + Number(item.score || 0), 0);
    return Math.round(total / history.length);
}

function getWeakPhonemes(history) {
    const countMap = {};

    history.forEach((item) => {
        item.phonemes?.forEach((phoneme) => {
            if (!phoneme.correct) {
                countMap[phoneme.symbol] = (countMap[phoneme.symbol] || 0) + 1;
            }
        });
    });

    return Object.entries(countMap)
        .map(([symbol, count]) => ({ symbol, count }))
        .sort((a, b) => b.count - a.count);
}

function getImprovedPhonemes(history) {
    const correctMap = {};

    history.forEach((item) => {
        item.phonemes?.forEach((phoneme) => {
            if (phoneme.correct) {
                correctMap[phoneme.symbol] = (correctMap[phoneme.symbol] || 0) + 1;
            }
        });
    });

    return Object.entries(correctMap)
        .map(([symbol, count]) => ({ symbol, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5);
}

export default function ProgressPage() {
    const [history, setHistory] = useState([]);

    useEffect(() => {
        setHistory(getHistory());
    }, []);

    const averageScore = useMemo(() => getAverageScore(history), [history]);
    const weakPhonemes = useMemo(() => getWeakPhonemes(history), [history]);
    const improvedPhonemes = useMemo(
        () => getImprovedPhonemes(history),
        [history]
    );

    const latestScore = history[0]?.score || 0;
    const previousScore = history[1]?.score || 0;
    const scoreDiff = latestScore - previousScore;

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8 text-slate-900">
            <div className="mx-auto max-w-6xl">
                <header className="mb-8">
                    <p className="text-sm font-extrabold uppercase text-purple-500">
                        Learning Progress
                    </p>

                    <h1 className="mt-2 text-4xl font-extrabold">
                        Theo dõi tiến bộ phát âm
                    </h1>

                    <p className="mt-2 text-slate-500">
                        Tổng hợp điểm số, lỗi âm vị thường gặp và gợi ý luyện tiếp.
                    </p>
                </header>

                {history.length === 0 ? (
                    <section className="rounded-3xl bg-white p-10 text-center shadow-sm">
                        <h2 className="text-2xl font-extrabold">Chưa có dữ liệu luyện tập</h2>
                        <p className="mt-3 text-slate-500">
                            Hãy luyện phát âm ít nhất một lần để xem tiến bộ.
                        </p>
                    </section>
                ) : (
                    <div className="space-y-8">
                        <section className="rounded-3xl bg-purple-600 p-8 text-white shadow-sm">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-lg font-extrabold">
                                        {scoreDiff >= 0
                                            ? `Bạn cải thiện +${scoreDiff} điểm so với lần trước`
                                            : `Bạn giảm ${Math.abs(scoreDiff)} điểm so với lần trước`}
                                    </p>

                                    <p className="mt-2 text-purple-100">
                                        AI khuyên bạn tập trung vào các âm vị sai nhiều nhất.
                                    </p>
                                </div>

                                <div className="rounded-2xl bg-white px-6 py-4 text-center text-purple-700">
                                    <p className="text-xs font-extrabold uppercase">
                                        Avg Score
                                    </p>
                                    <p className="text-4xl font-extrabold">{averageScore}</p>
                                </div>
                            </div>
                        </section>

                        <section className="grid grid-cols-3 gap-6">
                            <div className="rounded-3xl bg-white p-6 shadow-sm">
                                <p className="text-sm font-bold text-slate-400">
                                    Sessions Completed
                                </p>
                                <p className="mt-3 text-4xl font-extrabold">
                                    {history.length}
                                </p>
                            </div>

                            <div className="rounded-3xl bg-white p-6 shadow-sm">
                                <p className="text-sm font-bold text-slate-400">
                                    Latest Score
                                </p>
                                <p
                                    className={`mt-3 inline-block rounded-2xl px-5 py-2 text-4xl font-extrabold ${getScoreColor(
                                        latestScore
                                    )}`}
                                >
                                    {latestScore}
                                </p>
                            </div>

                            <div className="rounded-3xl bg-white p-6 shadow-sm">
                                <p className="text-sm font-bold text-slate-400">
                                    Weakest Phoneme
                                </p>
                                <p className="mt-3 text-4xl font-extrabold text-red-600">
                                    {weakPhonemes[0]?.symbol || "None"}
                                </p>
                            </div>
                        </section>

                        <section className="grid grid-cols-[1fr_360px] gap-8">
                            <div className="rounded-3xl bg-white p-8 shadow-sm">
                                <div className="mb-8 flex items-center justify-between">
                                    <div>
                                        <h2 className="text-xl font-extrabold">
                                            Pronunciation Score Trend
                                        </h2>
                                        <p className="text-sm text-slate-400">
                                            Các lần luyện gần đây
                                        </p>
                                    </div>
                                </div>

                                <div className="flex h-72 items-end gap-4 rounded-3xl bg-purple-50 p-6">
                                    {history
                                        .slice()
                                        .reverse()
                                        .slice(-8)
                                        .map((item, index) => (
                                            <div
                                                key={`${item.id}-${index}`}
                                                className="flex flex-1 flex-col items-center"
                                            >
                                                <div
                                                    className="w-full rounded-t-2xl bg-purple-500 transition-all"
                                                    style={{
                                                        height: `${Math.max(item.score, 8)}%`,
                                                    }}
                                                    title={`${item.score}/100`}
                                                />
                                                <p className="mt-3 text-xs font-bold text-slate-400">
                                                    {item.score}
                                                </p>
                                            </div>
                                        ))}
                                </div>
                            </div>

                            <div className="rounded-3xl bg-white p-6 shadow-sm">
                                <h2 className="text-xl font-extrabold">AI Recommendation</h2>

                                <p className="mt-3 text-sm leading-6 text-slate-500">
                                    Dựa trên lịch sử luyện tập, bạn nên tập trung luyện các âm vị
                                    đang xuất hiện lỗi nhiều nhất.
                                </p>

                                <div className="mt-6 space-y-3">
                                    {weakPhonemes.slice(0, 3).map((item) => (
                                        <div
                                            key={item.symbol}
                                            className="rounded-2xl bg-red-50 p-4"
                                        >
                                            <p className="text-2xl font-extrabold text-red-600">
                                                {item.symbol}
                                            </p>
                                            <p className="text-sm font-medium text-red-500">
                                                Sai {item.count} lần
                                            </p>
                                        </div>
                                    ))}
                                </div>

                                <button className="mt-6 w-full rounded-2xl bg-purple-600 py-4 font-extrabold text-white">
                                    Start Focus Practice
                                </button>
                            </div>
                        </section>

                        <section className="grid grid-cols-2 gap-8">
                            <div className="rounded-3xl bg-white p-6 shadow-sm">
                                <h2 className="text-xl font-extrabold text-emerald-700">
                                    Most Improved / Correct Sounds
                                </h2>

                                <div className="mt-5 space-y-3">
                                    {improvedPhonemes.map((item) => (
                                        <div
                                            key={item.symbol}
                                            className="flex items-center justify-between rounded-2xl bg-emerald-50 p-4"
                                        >
                                            <p className="text-2xl font-extrabold text-emerald-600">
                                                {item.symbol}
                                            </p>
                                            <p className="text-sm font-bold text-emerald-500">
                                                Correct {item.count} times
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="rounded-3xl bg-white p-6 shadow-sm">
                                <h2 className="text-xl font-extrabold text-red-700">
                                    Needs Practice
                                </h2>

                                <div className="mt-5 space-y-3">
                                    {weakPhonemes.length === 0 ? (
                                        <p className="rounded-2xl bg-emerald-50 p-4 font-bold text-emerald-600">
                                            Chưa có âm vị sai. Làm tốt lắm!
                                        </p>
                                    ) : (
                                        weakPhonemes.map((item) => (
                                            <div
                                                key={item.symbol}
                                                className="flex items-center justify-between rounded-2xl bg-red-50 p-4"
                                            >
                                                <p className="text-2xl font-extrabold text-red-600">
                                                    {item.symbol}
                                                </p>
                                                <p className="text-sm font-bold text-red-500">
                                                    {item.count} errors
                                                </p>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </section>
                    </div>
                )}
            </div>
        </main>
    );
}