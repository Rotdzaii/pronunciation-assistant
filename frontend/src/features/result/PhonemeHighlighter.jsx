import { useState } from "react";
import PhonemeDetailCard from "./PhonemeDetailCard";

export default function PhonemeHighlighter({ phonemes }) {
    const [selected, setSelected] = useState(null);

    return (
        <div className="grid grid-cols-[1fr_300px] gap-6">
            <section className="rounded-3xl bg-white p-8 shadow-sm">
                <p className="text-sm font-extrabold uppercase text-purple-500">
                    Phoneme Analysis
                </p>

                <div className="mt-6 flex flex-wrap gap-4">
                    {phonemes.map((phoneme, index) => (
                        <button
                            key={index}
                            onClick={() => setSelected(phoneme)}
                            className={`rounded-2xl px-5 py-3 text-lg font-extrabold transition ${phoneme.correct
                                    ? "bg-emerald-50 text-emerald-700"
                                    : "bg-red-50 text-red-700 ring-2 ring-red-200"
                                } hover:scale-105`}
                        >
                            {phoneme.symbol}
                        </button>
                    ))}
                </div>

                <p className="mt-6 text-sm text-slate-500">
                    Click vào từng âm để xem chi tiết cách phát âm.
                </p>
            </section>

            <PhonemeDetailCard phoneme={selected} />
        </div>
    );
}