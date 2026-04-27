const allowedTypes = [
    "audio/mpeg",
    "audio/wav",
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
];

export default function AudioUploader({ onAudioReady, disabled }) {
    function handleChange(event) {
        const file = event.target.files?.[0];

        if (!file) return;

        if (!allowedTypes.includes(file.type)) {
            alert("File audio không hợp lệ. Hãy chọn mp3, wav, webm, mp4 hoặc ogg.");
            return;
        }

        onAudioReady(file);
    }

    return (
        <label className="cursor-pointer rounded-3xl bg-white px-8 py-5 text-center font-bold shadow-sm transition hover:bg-purple-50">
            📁 Upload
            <input
                type="file"
                accept="audio/*"
                hidden
                disabled={disabled}
                onChange={handleChange}
            />
        </label>
    );
}