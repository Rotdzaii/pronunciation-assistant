export default function SelectableErrorItem({ item, selected, onToggle }) {
    return (
        <button
            type="button"
            onClick={() => onToggle(item.id)}
            className={`w-full rounded-2xl border p-4 text-left transition ${selected
                    ? "border-purple-500 bg-purple-50 shadow-sm"
                    : "border-slate-200 bg-white hover:border-purple-200 hover:bg-purple-50/40"
                }`}
        >
            <div className="flex items-start justify-between gap-4">
                <div>
                    <p className="text-lg font-extrabold text-slate-900">{item.text}</p>

                    <p className="mt-1 text-sm text-slate-500">{item.note}</p>

                    <div className="mt-3 flex flex-wrap gap-2">
                        <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-bold text-red-600">
                            Error: {item.errorType}
                        </span>

                        <span className="rounded-full bg-purple-50 px-3 py-1 text-xs font-bold text-purple-600">
                            Target: {item.targetPhoneme || "Sentence focus"}
                        </span>

                        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-600">
                            Previous score: {item.previousScore}
                        </span>
                    </div>
                </div>

                <div
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-sm font-extrabold ${selected
                            ? "border-purple-600 bg-purple-600 text-white"
                            : "border-slate-300 text-slate-300"
                        }`}
                >
                    {selected ? "✓" : ""}
                </div>
            </div>
        </button>
    );
}