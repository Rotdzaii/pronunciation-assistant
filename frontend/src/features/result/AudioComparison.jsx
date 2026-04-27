export default function AudioComparison({
    userAudioUrl,
    referenceAudioUrl,
}) {
    return (
        <section className="rounded-3xl bg-purple-50 p-6">
            <p className="text-sm font-extrabold uppercase text-purple-600">
                Listen & Compare
            </p>

            <div className="mt-6 space-y-4">
                {/* USER AUDIO */}
                <div className="rounded-2xl bg-white p-4">
                    <p className="text-xs font-bold text-slate-400">
                        YOUR PRONUNCIATION
                    </p>

                    {userAudioUrl ? (
                        <audio
                            controls
                            src={userAudioUrl}
                            className="mt-3 w-full"
                        />
                    ) : (
                        <p className="mt-3 text-sm text-red-400">
                            No user audio available
                        </p>
                    )}
                </div>

                {/* CORRECT AUDIO */}
                <div className="rounded-2xl bg-white p-4">
                    <p className="text-xs font-bold text-slate-400">
                        CORRECT PRONUNCIATION
                    </p>

                    {referenceAudioUrl ? (
                        <audio
                            controls
                            src={referenceAudioUrl}
                            className="mt-3 w-full"
                        />
                    ) : (
                        <p className="mt-3 text-sm text-slate-400">
                            Reference audio coming soon
                        </p>
                    )}
                </div>
            </div>

            <div className="mt-5 rounded-2xl bg-white p-4 text-sm text-slate-500">
                💡 Nghe và so sánh để nhận ra sự khác biệt giữa phát âm của bạn và
                phát âm chuẩn.
            </div>
        </section>
    );
}