import React, { useMemo } from 'react';
import { 
  BarChart as ReBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { usePracticeHistory } from '../lib/usePracticeHistory';

export const AnalyticsScreen = () => {
    const { history, loading, error } = usePracticeHistory();
    const data = useMemo(() => {
        const grouped = new Map<string, number[]>();
        history.forEach((item) => {
            const date = new Date(item.created_at);
            const key = date.toLocaleDateString('vi-VN', { weekday: 'short' });
            if (!grouped.has(key)) {
                grouped.set(key, []);
            }
            grouped.get(key)?.push(item.overall_score);
        });

        const entries = Array.from(grouped.entries()).map(([name, scores]) => {
            const avg = scores.reduce((sum, val) => sum + val, 0) / scores.length;
            return { name, score: Math.round(avg) };
        });

        return entries.slice(-7);
    }, [history]);

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500 pb-12">
        <header>
            <h2 className="text-3xl font-bold tracking-tight">Tiến độ của bạn</h2>
            <p className="text-on-surface-variant mt-2">Theo dõi tiến bộ qua từng ngày luyện tập.</p>
        </header>

        <section className="bg-white p-6 rounded-[32px] border border-outline-variant/30 shadow-sm">
            <h3 className="font-bold text-lg mb-6">Xu hướng điểm số</h3>
            <div className="h-64 w-full">
                {loading ? (
                    <div className="h-full flex items-center justify-center text-sm text-on-surface-variant">
                        Đang tải dữ liệu...
                    </div>
                ) : data.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-sm text-on-surface-variant">
                        Chưa có dữ liệu điểm số.
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <ReBarChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748B' }} dy={10} />
                            <YAxis hide />
                            <Tooltip 
                                cursor={{ fill: 'transparent' }}
                                content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        return (
                                            <div className="bg-on-surface text-white px-2 py-1 rounded text-xs font-bold shadow-lg">
                                                {payload[0].value}%
                                            </div>
                                        );
                                    }
                                    return null;
                                }}
                            />
                            <Bar dataKey="score" radius={[8, 8, 0, 0]}>
                                {data.map((entry, index) => (
                                    <Cell 
                                        key={entry.name}
                                        fill={index === data.length - 1 ? '#004AC6' : '#2563EB'} 
                                        fillOpacity={0.2 + (index * 0.1)} 
                                    />
                                ))}
                            </Bar>
                        </ReBarChart>
                    </ResponsiveContainer>
                )}
            </div>
        </section>

        <section className="bg-white p-6 rounded-[32px] border border-outline-variant/30 shadow-sm">
            <h3 className="font-bold text-lg mb-4">Lịch sử luyện tập</h3>
            {error ? (
                <p className="text-sm text-error">{error}</p>
            ) : history.length === 0 ? (
                <p className="text-sm text-on-surface-variant">Chưa có dữ liệu luyện tập.</p>
            ) : (
                <div className="flex flex-col gap-4">
                    {history.slice(0, 5).map(item => (
                        <div key={item.id ?? item.created_at} className="flex items-center gap-4 p-4 rounded-2xl bg-surface-container-low">
                            <div className="w-10 h-10 rounded-full bg-primary-container/20 flex items-center justify-center text-primary font-bold">
                                {Math.round(item.overall_score)}
                            </div>
                            <div className="flex-1 gap-1 flex flex-col">
                                <div className="text-sm font-bold text-on-surface">{item.target_word}</div>
                                <div className="text-xs text-on-surface-variant">
                                    {new Date(item.created_at).toLocaleString('vi-VN')}
                                </div>
                            </div>
                            <div className="text-xs font-bold text-primary">{item.phoneme_details?.length ?? 0} âm</div>
                        </div>
                    ))}
                </div>
            )}
        </section>
    </div>
  );
};
