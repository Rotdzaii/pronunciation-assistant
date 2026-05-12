import React, { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, Flame, Lightbulb } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { usePracticeHistory } from '../lib/usePracticeHistory';

export const ProfileScreen = () => {
    const { history, stats } = usePracticeHistory();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [authError, setAuthError] = useState<string | null>(null);
    const [authLoading, setAuthLoading] = useState(false);
    const [userEmail, setUserEmail] = useState<string | null>(null);

    useEffect(() => {
        const loadUser = async () => {
            const { data } = await supabase.auth.getUser();
            setUserEmail(data.user?.email ?? null);
        };

        void loadUser();

        const { data } = supabase.auth.onAuthStateChange((_event, session) => {
            setUserEmail(session?.user?.email ?? null);
        });

        return () => {
            data.subscription.unsubscribe();
        };
    }, []);

    const averageScore = useMemo(() => {
        if (history.length === 0) {
            return null;
        }
        const sum = history.reduce((total, item) => total + item.overall_score, 0);
        return Math.round(sum / history.length);
    }, [history]);

    const handleSignIn = async () => {
        setAuthLoading(true);
        setAuthError(null);
        const { error } = await supabase.auth.signInWithPassword({
            email,
            password,
        });
        if (error) {
            setAuthError(error.message);
        }
        setAuthLoading(false);
    };

    const handleSignOut = async () => {
        setAuthLoading(true);
        setAuthError(null);
        const { error } = await supabase.auth.signOut();
        if (error) {
            setAuthError(error.message);
        }
        setAuthLoading(false);
    };

    return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500 pb-12">
        <header className="flex items-center gap-4">
            <button className="p-2 hover:bg-surface-container rounded-full"><ArrowLeft size={20} /></button>
            <h2 className="text-xl font-bold uppercase tracking-widest text-on-surface-variant">Chi tiết học viên</h2>
        </header>

        <div className="bg-white p-8 rounded-[32px] border border-outline-variant/30 shadow-sm flex flex-col items-center text-center">
            <div className="w-24 h-24 rounded-full overflow-hidden mb-4 border-4 border-surface-container shadow-sm relative">
                <img 
                    src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=1976&auto=format&fit=crop" 
                    alt="Learner"
                    className="w-full h-full object-cover"
                />
            </div>
            <h3 className="text-2xl font-bold text-on-surface">
                {userEmail ?? 'Chưa đăng nhập'}
            </h3>
            <p className="text-sm text-on-surface-variant">
                {userEmail ? 'Đang đồng bộ lịch sử luyện tập' : 'Vui lòng đăng nhập để đồng bộ dữ liệu'}
            </p>
            
            <div className="w-full grid grid-cols-2 gap-4 border-t border-outline-variant/30 pt-6 mt-6">
                <div className="flex flex-col items-center">
                    <span className="text-2xl font-bold text-primary">
                        {averageScore === null ? '--' : `${averageScore}%`}
                    </span>
                    <span className="text-[10px] text-on-surface-variant uppercase font-bold">Độ chính xác</span>
                </div>
                <div className="flex flex-col items-center border-l border-outline-variant/30">
                    <span className="text-2xl font-bold text-tertiary flex items-center gap-1">
                        {stats.streak} <Flame size={20} fill="currentColor" />
                    </span>
                    <span className="text-[10px] text-on-surface-variant uppercase font-bold">Ngày liên tiếp</span>
                </div>
            </div>

            {!userEmail && (
                <div className="w-full mt-6 flex flex-col gap-3 text-left">
                    <input
                        value={email}
                        onChange={(event) => setEmail(event.target.value)}
                        type="email"
                        placeholder="Email"
                        className="w-full rounded-xl border border-outline-variant/40 px-4 py-3 text-sm"
                    />
                    <input
                        value={password}
                        onChange={(event) => setPassword(event.target.value)}
                        type="password"
                        placeholder="Mật khẩu"
                        className="w-full rounded-xl border border-outline-variant/40 px-4 py-3 text-sm"
                    />
                    {authError && <p className="text-sm text-error">{authError}</p>}
                    <button
                        onClick={handleSignIn}
                        className="w-full h-12 bg-primary text-white font-bold rounded-xl"
                        disabled={authLoading}
                    >
                        {authLoading ? 'Đang đăng nhập...' : 'Đăng nhập'}
                    </button>
                </div>
            )}

            {userEmail && (
                <button
                    onClick={handleSignOut}
                    className="mt-6 w-full h-12 border border-primary text-primary font-bold rounded-xl"
                    disabled={authLoading}
                >
                    {authLoading ? 'Đang đăng xuất...' : 'Đăng xuất'}
                </button>
            )}
        </div>

        <section className="bg-secondary-container/10 border border-secondary/20 rounded-3xl p-6 relative overflow-hidden">
            <div className="flex items-start gap-4 relative z-10">
                <div className="w-12 h-12 bg-secondary-container rounded-full flex items-center justify-center shrink-0">
                    <Lightbulb className="text-on-secondary-container" size={24} />
                </div>
                <div>
                    <h3 className="font-bold text-lg text-on-secondary-container">Chiến lược can thiệp AI</h3>
                    <p className="text-sm text-on-surface-variant mt-2 leading-relaxed">
                        Độ chính xác với âm /θ/ giảm 15%. Đề xuất bài tập "Vị trí đặt lưỡi".
                    </p>
                    <button className="mt-4 bg-secondary text-white font-bold py-3 px-6 rounded-xl shadow-md text-sm">
                        Giao bài tập mục tiêu
                    </button>
                </div>
            </div>
        </section>

        <section className="bg-white p-6 rounded-[32px] border border-outline-variant/30">
            <h3 className="font-bold text-lg mb-4">Lỗi thường gặp</h3>
            <div className="flex flex-wrap gap-2">
                <span className="bg-error-container text-on-error-container px-3 py-1.5 rounded-full text-xs font-bold">/θ/ (th)</span>
                <span className="bg-tertiary-container text-on-tertiary-container px-3 py-1.5 rounded-full text-xs font-bold">/v/ vs /b/</span>
                <span className="bg-surface-container text-on-surface-variant px-3 py-1.5 rounded-full text-xs font-bold">/r/ ending</span>
            </div>
        </section>
    </div>
    );
};
