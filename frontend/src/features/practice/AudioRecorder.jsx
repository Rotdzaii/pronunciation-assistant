import { useRef, useState } from "react";

export default function AudioRecorder({ onAudioReady, disabled }) {
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const [isRecording, setIsRecording] = useState(false);
    const [error, setError] = useState("");

    async function startRecording() {
        try {
            setError("");
            chunksRef.current = [];

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);

            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(chunksRef.current, {
                    type: "audio/webm",
                });

                stream.getTracks().forEach((track) => track.stop());
                setIsRecording(false);
                onAudioReady(audioBlob);
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch {
            setError("Không thể truy cập microphone. Hãy kiểm tra quyền trình duyệt.");
            setIsRecording(false);
        }
    }

    function stopRecording() {
        mediaRecorderRef.current?.stop();
    }

    return (
        <div className="flex flex-col items-center gap-4">
            <button
                type="button"
                disabled={disabled}
                onClick={isRecording ? stopRecording : startRecording}
                className={`flex h-40 w-40 items-center justify-center rounded-full border-8 border-white text-white shadow-xl transition ${isRecording ? "animate-pulse bg-red-500" : "bg-purple-500"
                    } disabled:cursor-not-allowed disabled:opacity-50`}
            >
                <div className="text-center">
                    <div className="text-5xl">🎙️</div>
                    <p className="mt-2 text-xs font-bold uppercase">
                        {isRecording ? "Tap to stop" : "Tap to start"}
                    </p>
                </div>
            </button>

            {isRecording && (
                <p className="text-sm font-bold text-red-500">Recording...</p>
            )}

            {error && <p className="text-sm text-red-500">{error}</p>}
        </div>
    );
}