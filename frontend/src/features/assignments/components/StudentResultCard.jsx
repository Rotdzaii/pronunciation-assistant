export default function StudentResultCard({ item }) {
    const hasResult =
        typeof item.latestScore === "number" &&
        typeof item.previousScore === "number";

    return (
        <div className="rounded-2xl border border-slate-200 p-5 bg-white">
            <div className="flex justify-between items-start">
                <div>
                    <p className="text-lg font-extrabold">{item.text}</p>

                    <p className="text-sm text-slate-500 mt-1">
                        Target: {item.targetPhoneme || "Sentence"}
                    </p>
                </div>

                {hasResult ? (
                    <div className="text-right">
                        <p className="text-xs text-slate-400">Before → After</p>

                        <p className="text-xl font-extrabold">
                            {item.previousScore} →{" "}
                            <span className="text-purple-600">{item.latestScore}</span>
                        </p>
                    </div>
                ) : (
                    <p className="text-sm text-slate-400">Not submitted</p>
                )}
            </div>

            {hasResult && (
                <p
                    className={`mt-3 text-sm font-bold ${item.latestScore >= item.previousScore
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                >
                    {item.latestScore >= item.previousScore
                        ? `Improved +${item.latestScore - item.previousScore}`
                        : `Dropped ${item.previousScore - item.latestScore}`}
                </p>
            )}
        </div>
    );
}