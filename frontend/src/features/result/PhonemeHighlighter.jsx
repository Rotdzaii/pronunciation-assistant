export default function PhonemeHighlighter({ phonemes }) {
    return (
        <section className="rounded-3xl bg-white p-8 shadow-sm">
            <p className="text-sm font-extrabold uppercase text-purple-500">
                Phoneme Analysis
            </p>

            <h2 className="mt-3 text-2xl font-extrabold text-slate-900">
                Âm vị trong từ
            </h2>

            <div className="mt-6 flex flex-wrap gap-4">
                {phonemes.map((phoneme) => (
                    <div
                        key={phoneme.symbol}
                        className={`rounded-2xl border px-6 py-4 text-xl font-extrabold ${phoneme.correct
                                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                : "border-red-200 bg-red-50 text-red-700 ring-2 ring-red-100"
                            }`}
                    >
                        <span className="mr-2">{phoneme.correct ? "✓" : "!"}</span>
                        {phoneme.symbol}
                    </div>
                ))}
            </div>

            <div className="mt-6 grid grid-cols-2 gap-4 text-sm font-bold">
                <div className="rounded-2xl bg-emerald-50 p-4 text-emerald-700">
                    Xanh = phát âm đúng
                </div>

                <div className="rounded-2xl bg-red-50 p-4 text-red-700">
                    Đỏ = âm vị cần sửa
                </div>
            </div>
        </section>
    );
}