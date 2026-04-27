export default function PronunciationScore({ score }) {
    function getLevel() {
        if (score >= 85) return "Excellent";
        if (score >= 70) return "Good";
        return "Needs Practice";
    }

    function getColorClass() {
        if (score >= 85) return "bg-emerald-500";
        if (score >= 70) return "bg-yellow-400";
        return "bg-red-500";
    }

    function getTextColorClass() {
        if (score >= 85) return "text-emerald-600";
        if (score >= 70) return "text-yellow-600";
        return "text-red-600";
    }

    return (
        <section className="rounded-3xl bg-white p-8 shadow-sm">
            <p className="text-sm font-extrabold uppercase text-purple-500">
                Pronunciation Score
            </p>

            <div className="mt-6 flex items-end gap-3">
                <h1 className={`text-7xl font-extrabold ${getTextColorClass()}`}>
                    {score}
                </h1>
                <p className="mb-3 text-xl font-bold text-slate-400">/100</p>
            </div>

            <p className={`mt-3 text-lg font-extrabold ${getTextColorClass()}`}>
                {getLevel()}
            </p>

            <div className="mt-6 h-4 overflow-hidden rounded-full bg-slate-100">
                <div
                    className={`h-full rounded-full transition-all ${getColorClass()}`}
                    style={{ width: `${score}%` }}
                />
            </div>
        </section>
    );
}