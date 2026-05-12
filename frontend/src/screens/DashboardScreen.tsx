import React from 'react';
import { Mic, Play, Flame, Timer, Lightbulb } from 'lucide-react';
import { usePracticeHistory } from '../lib/usePracticeHistory';

interface DashboardScreenProps {
  onPractice: () => void;
}

export const DashboardScreen = ({ onPractice }: DashboardScreenProps) => {
    const { stats, history } = usePracticeHistory();
    const latestScore = stats.latestScore;
    const trend = history.length >= 2
        ? Math.round((history[0].overall_score ?? 0) - (history[1].overall_score ?? 0))
        : null;
    const insight = latestScore === null
        ? 'Chưa có dữ liệu luyện tập. Bắt đầu một bài để xem phân tích.'
        : trend === null
            ? 'Hệ thống đã sẵn sàng phân tích bài nói tiếp theo của bạn.'
            : trend >= 0
                ? `Điểm gần nhất tăng ${trend}% so với lần trước.`
                : `Điểm gần nhất giảm ${Math.abs(trend)}% so với lần trước.`;

    return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500">
        <section>
            <h2 className="text-3xl font-bold text-on-surface tracking-tight">Chào buổi sáng,</h2>
            <p className="text-lg text-on-surface-variant font-medium">Hôm nay luyện 5 phút nhé?</p>
        </section>

        <button 
            onClick={onPractice}
            className="w-full bg-gradient-to-br from-primary to-primary-container p-8 rounded-[32px] text-white shadow-xl relative overflow-hidden group text-left"
        >
            <div className="absolute -right-8 -top-8 w-40 h-40 bg-white/10 rounded-full blur-2xl group-hover:bg-white/20 transition-all" />
            <div className="relative z-10 flex items-center justify-between">
                <div>
                    <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center mb-4">
                        <Mic size={20} />
                    </div>
                    <h3 className="text-2xl font-bold">Luyện phát âm ngay</h3>
                    <p className="text-sm text-white/80 mt-1">Bắt đầu bài học đầu tiên</p>
                </div>
                <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-primary shadow-lg">
                    <Play fill="currentColor" size={24} />
                </div>
            </div>
        </button>

        <div className="grid grid-cols-3 gap-3">
            <div className="bg-white p-4 rounded-2xl border border-outline-variant/30 flex flex-col items-center">
                <div className="w-10 h-10 rounded-full bg-surface-container flex items-center justify-center mb-2">
                    <span className="text-xs font-bold text-on-surface-variant">
                        {latestScore === null ? '--' : `${Math.round(latestScore)}%`}
                    </span>
                </div>
                <span className="text-[10px] text-on-surface-variant font-medium">Điểm gần nhất</span>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-outline-variant/30 flex flex-col items-center">
                <Flame className="text-tertiary mb-2" size={20} fill="currentColor" />
                <span className="text-lg font-bold">{stats.streak}</span>
                <span className="text-[10px] text-on-surface-variant font-medium">Ngày chuỗi</span>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-outline-variant/30 flex flex-col items-center">
                <Timer className="text-secondary mb-2" size={20} />
                <span className="text-lg font-bold">{stats.totalAttempts}</span>
                <span className="text-[10px] text-on-surface-variant font-medium">Số buổi</span>
            </div>
        </div>

        <section className="bg-white p-6 rounded-3xl border border-outline-variant/30 shadow-sm flex gap-4 items-start">
            <div className="w-10 h-10 bg-primary-container/20 rounded-full flex items-center justify-center shrink-0">
                <Lightbulb size={20} className="text-primary" />
            </div>
            <div>
                <h4 className="font-bold text-sm">Bản tin AI</h4>
                <p className="text-sm text-on-surface-variant mt-1 leading-relaxed">
                    {insight}
                </p>
            </div>
        </section>
    </div>
    );
};
