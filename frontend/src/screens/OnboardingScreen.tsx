import React from 'react';
import { Mic, Bolt } from 'lucide-react';
import { motion } from 'motion/react';

interface OnboardingScreenProps {
  onStart: () => void;
}

export const OnboardingScreen = ({ onStart }: OnboardingScreenProps) => (
    <div className="min-h-screen flex items-center justify-center p-4">
        <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-[500px] bg-white rounded-[32px] shadow-xl border border-outline-variant/30 overflow-hidden flex flex-col"
        >
            <div className="h-64 sm:h-72 relative bg-surface-container-low overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent" />
                <img 
                    src="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=2070&auto=format&fit=crop" 
                    className="w-full h-full object-cover mix-blend-multiply opacity-60"
                    alt="Learning"
                />
                <div className="absolute bottom-0 left-0 w-full p-8 bg-white/60 backdrop-blur-md border-t border-white/20">
                    <h1 className="text-2xl font-bold text-primary leading-tight">AI Pronunciation Coach for Vietnamese Learners</h1>
                    <p className="text-sm text-on-surface-variant mt-2">Luyện phát âm tiếng Anh, nhận góp ý bằng tiếng Việt.</p>
                </div>
            </div>
            
            <div className="p-8 flex flex-col gap-8">
                <div className="space-y-6">
                    <div className="flex gap-4">
                        <div className="w-12 h-12 rounded-full bg-primary-container flex items-center justify-center shrink-0">
                            <Mic className="text-primary" size={20} />
                        </div>
                        <div>
                            <span className="text-[10px] uppercase font-bold text-primary tracking-widest">Bước 1</span>
                            <h3 className="font-bold text-on-surface">Ghi âm giọng đọc</h3>
                            <p className="text-sm text-on-surface-variant">Đọc các câu mẫu để hệ thống ghi nhận.</p>
                        </div>
                    </div>
                    <div className="flex gap-4">
                        <div className="w-12 h-12 rounded-full bg-secondary-container flex items-center justify-center shrink-0">
                            <Bolt className="text-secondary" size={20} />
                        </div>
                        <div>
                            <span className="text-[10px] uppercase font-bold text-secondary tracking-widest">Bước 2</span>
                            <h3 className="font-bold text-on-surface">AI phát hiện lỗi</h3>
                            <p className="text-sm text-on-surface-variant">Phân tích từng âm vị và đánh dấu lỗi.</p>
                        </div>
                    </div>
                </div>

                <div className="flex gap-4">
                    <button 
                        onClick={onStart}
                        className="flex-1 h-14 bg-primary text-white font-bold rounded-2xl shadow-lg hover:shadow-primary/20 transition-all"
                    >
                        Bắt đầu ngay
                    </button>
                    <button className="flex-1 h-14 border border-primary text-primary font-bold rounded-2xl hover:bg-primary/5 transition-all">
                        Tôi là giáo viên
                    </button>
                </div>
            </div>
        </motion.div>
    </div>
);
