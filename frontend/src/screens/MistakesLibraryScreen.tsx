import React, { useMemo } from 'react';
import { AlertCircle, ArrowRight, Mic } from 'lucide-react';
import { cn } from '../lib/utils';
import { usePracticeHistory } from '../lib/usePracticeHistory';

export const MistakesLibraryScreen = () => {
    const { history, loading, error } = usePracticeHistory();
    const mistakes = useMemo(() => {
        const map = new Map<string, { scoreSum: number; count: number; ipa: string }>();
        history.forEach((item) => {
            item.phoneme_details?.forEach((detail) => {
                const key = detail.phoneme;
                const existing = map.get(key);
                if (existing) {
                    existing.scoreSum += detail.score;
                    existing.count += 1;
                } else {
                    map.set(key, {
                        scoreSum: detail.score,
                        count: 1,
                        ipa: detail.ipa,
                    });
                }
            });
        });

        return Array.from(map.entries())
            .map(([phoneme, data], index) => ({
                id: index,
                phoneme,
                ipa: data.ipa,
                score: data.scoreSum / data.count,
            }))
            .sort((a, b) => a.score - b.score)
            .slice(0, 3)
            .map((item, index) => ({
                id: item.id,
                title: `Âm ${item.phoneme}`,
                desc: `Điểm trung bình ${Math.round(item.score)}% - IPA ${item.ipa}`,
                color:
                    index === 0
                        ? 'bg-error-container text-on-error-container'
                        : index === 1
                            ? 'bg-tertiary-container text-on-tertiary-container'
                            : 'bg-primary-container text-on-primary-container',
                wrong: item.phoneme,
                right: item.ipa,
            }));
    }, [history]);

    return (
        <div className="flex flex-col gap-8 animate-in fade-in duration-500 pb-12">
            <header>
                <h2 className="text-3xl font-bold tracking-tight">Thư viện lỗi phổ biến</h2>
                <p className="text-on-surface-variant mt-2">Khắc phục các lỗi phát âm đặc trưng của người Việt.</p>
            </header>

            <div className="flex flex-col gap-4">
                {loading ? (
                    <div className="text-sm text-on-surface-variant">Đang tải dữ liệu...</div>
                ) : error ? (
                    <div className="text-sm text-error">{error}</div>
                ) : mistakes.length === 0 ? (
                    <div className="text-sm text-on-surface-variant">Chưa có lỗi phát âm nào.</div>
                ) : (
                    mistakes.map(item => (
                        <div key={item.id} className="bg-white p-6 rounded-[24px] border border-outline-variant/30 shadow-sm flex flex-col gap-6">
                            <div className="flex gap-4 items-start">
                                <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center shrink-0", item.color)}>
                                    <AlertCircle size={24} />
                                </div>
                                <div>
                                    <h3 className="text-xl font-bold">{item.title}</h3>
                                    <p className="text-sm text-on-surface-variant mt-1">{item.desc}</p>
                                </div>
                            </div>
                            <div className="bg-surface-container-low p-4 rounded-xl flex justify-between items-center text-sm">
                                <span className="text-error line-through">{item.wrong}</span>
                                <ArrowRight size={16} className="text-outline-variant" />
                                <span className="text-primary font-bold">{item.right}</span>
                            </div>
                            <button className="w-full h-14 bg-primary text-white font-bold rounded-2xl flex items-center justify-center gap-2 shadow-md">
                                <Mic size={20} /> Luyện tập ngay
                            </button>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
