export default function PhonemeDetailCard({ phoneme }) {
    if (!phoneme) {
        return (
            <div className="rounded-2xl bg-slate-50 p-6 text-center text-slate-400">
                Click vào một phoneme để xem chi tiết
            </div>
        );
    }

    return (
        <div className="rounded-3xl bg-white p-6 shadow-sm">
            <p className="text-sm font-extrabold uppercase text-purple-500">
                Phoneme Detail
            </p>

            <h2 className="mt-4 text-4xl font-extrabold">
                {phoneme.symbol}
            </h2>

            <div className="mt-6 space-y-3">
                <div className="rounded-2xl bg-emerald-50 p-4">
                    <p className="text-xs font-bold text-emerald-500">
                        EXPECTED
                    </p>
                    <p className="text-xl font-extrabold text-emerald-700">
                        {phoneme.symbol}
                    </p>
                </div>

                <div className="rounded-2xl bg-red-50 p-4">
                    <p className="text-xs font-bold text-red-500">
                        DETECTED
                    </p>
                    <p className="text-xl font-extrabold text-red-700">
                        {phoneme.correct ? phoneme.symbol : "Incorrect articulation"}
                    </p>
                </div>
            </div>

            {!phoneme.correct && (
                <div className="mt-5 rounded-2xl bg-yellow-50 p-4 text-sm text-yellow-700">
                    Hãy điều chỉnh lưỡi và luồng khí để cải thiện âm thanh này.
                </div>
            )}
        </div>
    );
}