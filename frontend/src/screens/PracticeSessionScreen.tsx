import React, { useEffect, useRef, useState } from 'react';
import { Flame, Volume2, Mic, ArrowRight } from 'lucide-react';
import { motion } from 'motion/react';
import { cn } from '../lib/utils';
import { uploadAudio } from '../lib/api';
import { usePracticeHistory } from '../lib/usePracticeHistory';
import type { AnalyzeResponse } from '../types';

interface PracticeSessionScreenProps {
  onComplete: () => void;
}

export const PracticeSessionScreen = ({ onComplete }: PracticeSessionScreenProps) => {
    const [recording, setRecording] = useState(false);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
    const [submitting, setSubmitting] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const { refresh } = usePracticeHistory();

    useEffect(() => {
        return () => {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [audioUrl]);

    const startRecording = async () => {
        setError(null);
        setAnalysis(null);
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream);

        chunksRef.current = [];
        recorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                chunksRef.current.push(event.data);
            }
        };
        recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
            setAudioBlob(blob);
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
            setAudioUrl(URL.createObjectURL(blob));
            stream.getTracks().forEach((track) => track.stop());
        };

        recorder.start();
        mediaRecorderRef.current = recorder;
        setRecording(true);
    };

    const stopRecording = () => {
        mediaRecorderRef.current?.stop();
        setRecording(false);
    };

    const toggleRecording = () => {
        if (recording) {
            stopRecording();
        } else {
            void startRecording();
        }
    };

    const handleSubmit = async () => {
        if (!audioBlob) {
            setError('Vui lòng ghi âm trước khi gửi.');
            return;
        }

        setSubmitting(true);
        setError(null);
        try {
            const file = new File(
                [audioBlob],
                `practice-${Date.now()}.webm`,
                { type: audioBlob.type || 'audio/webm' },
            );
            const result = await uploadAudio(file);
            setAnalysis(result);
            await refresh();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Không gửi được file.');
        } finally {
            setSubmitting(false);
        }
    };

    const playAudio = () => {
        if (audioRef.current) {
            void audioRef.current.play();
        }
    };

    const transcription = analysis?.ai_thinking?.step_2_stt ?? 'Chưa có kết quả';
    const phonemes = analysis?.phoneme_details ?? [];
    
    return (
        <div className="flex flex-col h-full animate-in zoom-in-95 duration-500 py-4">
            <div className="flex justify-between items-center mb-12">
                <div className="bg-surface-container px-3 py-1 rounded-full text-[10px] font-bold">Từ 3 trên 10</div>
                <div className="bg-tertiary-container text-on-tertiary-container px-3 py-1 rounded-full text-[10px] font-bold flex items-center gap-1">
                    <Flame size={12} fill="currentColor" /> Chuỗi 5 ngày
                </div>
            </div>

            <div className="bg-white rounded-[32px] p-10 shadow-sm border border-outline-variant/30 text-center flex flex-col items-center mb-12">
                <span className="text-[10px] uppercase font-bold text-on-surface-variant mb-6 tracking-widest">
                    {analysis ? 'Kết quả nhận diện' : 'Bài luyện hôm nay'}
                </span>

                <h2 className="text-3xl sm:text-4xl font-extrabold text-on-surface mb-2">
                    {analysis ? transcription : 'Hãy bắt đầu ghi âm'}
                </h2>

                <div className="flex items-center gap-3">
                    <span className="text-xl text-primary font-medium">
                        {analysis ? `${Math.round(analysis.overall_score)}%` : '--'}
                    </span>
                    <button
                        onClick={playAudio}
                        className="p-2 bg-surface-container rounded-full text-primary"
                        disabled={!audioUrl}
                    >
                        <Volume2 size={24} />
                    </button>
                </div>

                {analysis ? (
                    <div className="mt-6 flex flex-wrap justify-center gap-2">
                        {phonemes.length === 0 ? (
                            <span className="text-sm text-on-surface-variant">Chưa có chi tiết âm vị.</span>
                        ) : (
                            phonemes.slice(0, 10).map((item, index) => (
                                <span
                                    key={`${item.phoneme}-${index}`}
                                    className="px-2 py-1 rounded-lg bg-surface-container text-on-surface-variant font-bold text-xs"
                                >
                                    {item.phoneme} {item.ipa} ({Math.round(item.score)}%)
                                </span>
                            ))
                        )}
                    </div>
                ) : (
                    <p className="text-sm text-on-surface-variant italic mt-6">Ghi âm để hệ thống phân tích.</p>
                )}
            </div>

            <div className="mt-auto flex flex-col items-center gap-8">
                {recording ? (
                    <div className="flex flex-col items-center gap-4 animate-in fade-in slide-in-from-top-2">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-error animate-pulse" />
                            <span className="text-sm font-bold text-on-surface-variant">Đang ghi âm... 00:03</span>
                        </div>
                        <div className="flex gap-1 h-8 items-center">
                            {[2,4,6,3,5,2,4,2].map((h, i) => (
                                <motion.div 
                                    key={i}
                                    animate={{ height: [h*4, h*2, h*4] }}
                                    transition={{ repeat: Infinity, duration: 0.5, delay: i*0.1 }}
                                    className="w-1.5 bg-secondary rounded-full"
                                />
                            ))}
                        </div>
                    </div>
                ) : (
                    <p className="text-sm font-medium text-on-surface-variant">Nhấn Micro để bắt đầu nói</p>
                )}

                <button 
                    onClick={toggleRecording}
                    className={cn(
                        "w-20 h-20 rounded-full flex items-center justify-center transition-all shadow-lg active:scale-95",
                        recording ? "bg-surface-variant text-on-surface-variant" : "bg-primary text-white"
                    )}
                >
                    <Mic size={32} fill={recording ? "none" : "currentColor"} />
                </button>

                <div className="flex gap-4 w-full">
                    <button
                        onClick={playAudio}
                        className="flex-1 h-14 bg-surface-container text-on-surface font-bold rounded-2xl shadow-sm"
                        disabled={!audioUrl}
                    >
                        Nghe lại
                    </button>
                    <button 
                        onClick={handleSubmit}
                        className="flex-1 h-14 bg-primary text-white font-bold rounded-2xl shadow-lg flex items-center justify-center gap-2"
                        disabled={submitting}
                    >
                        {submitting ? 'Đang gửi...' : 'Gửi AI chấm'} <ArrowRight size={20} />
                    </button>
                </div>

                {error && (
                    <p className="text-sm text-error font-medium">{error}</p>
                )}

                {analysis && (
                    <button
                        onClick={onComplete}
                        className="text-sm font-bold text-outline-variant hover:text-on-surface transition-colors mb-4"
                    >
                        Hoàn tất phiên
                    </button>
                )}
            </div>
            <audio ref={audioRef} src={audioUrl ?? undefined} />
        </div>
    );
};
