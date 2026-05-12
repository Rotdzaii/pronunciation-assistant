import React from 'react';
import { Bolt, Type, AlignLeft, Timer } from 'lucide-react';
import { cn } from '../lib/utils';

interface PracticeMenuScreenProps {
  onStartSession: () => void;
}

export const PracticeMenuScreen = ({ onStartSession }: PracticeMenuScreenProps) => {
    const modes = [
        { 
            id: 1, 
            title: 'Lỗi người Việt hay gặp', 
            desc: 'Khắc phục triệt để các lỗi như /θ/, /ð/, âm đuôi.', 
            time: '15p',
            level: 'Trung bình',
            featured: true,
            icon: Bolt
        },
        { 
            id: 2, 
            title: 'Luyện từ đơn', 
            desc: 'Chuẩn xác từng âm tiết cơ bản.', 
            time: '5p',
            level: 'Dễ',
            icon: Type
        },
        { 
            id: 3, 
            title: 'Luyện câu ngắn', 
            desc: 'Luyện ngữ điệu và trọng âm câu.', 
            time: '10p',
            level: 'Vừa',
            icon: AlignLeft
        },
    ];

    return (
        <div className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-12">
            <header>
                <h2 className="text-3xl font-bold tracking-tight">Chế độ luyện tập</h2>
                <p className="text-on-surface-variant mt-2">Chọn phương pháp phù hợp với mục tiêu của bạn.</p>
            </header>

            <div className="flex flex-col gap-4">
                {modes.map(mode => (
                    <button
                        key={mode.id}
                        onClick={onStartSession}
                        className={cn(
                            "p-6 rounded-[24px] border transition-all text-left group relative overflow-hidden",
                            mode.featured 
                                ? "bg-secondary-container/40 border-secondary/20 shadow-sm" 
                                : "bg-white border-outline-variant/30 hover:border-primary/30"
                        )}
                    >
                        <div className="flex gap-6 items-center relative z-10">
                            <div className={cn(
                                "w-14 h-14 rounded-2xl flex items-center justify-center shrink-0",
                                mode.featured ? "bg-white text-secondary" : "bg-surface-container text-primary"
                            )}>
                                <mode.icon size={28} />
                            </div>
                            <div className="flex-1">
                                <h3 className="text-xl font-bold text-on-surface">{mode.title}</h3>
                                <p className="text-sm text-on-surface-variant mt-1">{mode.desc}</p>
                                <div className="flex gap-4 mt-3">
                                    <div className="flex items-center gap-1 text-[10px] bg-white/50 px-2 py-1 rounded-full text-on-surface-variant font-bold">
                                        <Timer size={12} /> {mode.time}
                                    </div>
                                    <div className="flex items-center gap-1 text-[10px] bg-white/50 px-2 py-1 rounded-full text-on-surface-variant font-bold">
                                         {mode.level}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
};
