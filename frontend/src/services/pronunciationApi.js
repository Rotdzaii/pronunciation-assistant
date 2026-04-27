const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const buildFormData = ({ audioBlob, transcript, assignmentId, itemId }) => {
    const formData = new FormData();

    formData.append("audio", audioBlob, "recording.webm");
    formData.append("transcript", transcript || "");

    if (assignmentId) {
        formData.append("assignmentId", assignmentId);
    }

    if (itemId) {
        formData.append("itemId", itemId);
    }

    return formData;
};

const normalizeResult = (result) => {
    return {
        score: result.score ?? 0,
        word: result.word || result.target || "",
        sentence: result.sentence || result.transcript || "",
        phonemes: result.phonemes || [],
        suggestion: result.suggestion || "Keep practicing and focus on weak sounds.",
        phonemeErrors: result.phonemeErrors || result.phoneme_errors || [],
    };
};

export const submitPronunciationAudio = async ({
    audioBlob,
    transcript,
    assignmentId = null,
    itemId = null,
}) => {
    const response = await fetch(`${API_BASE_URL}/api/pronunciation/analyze`, {
        method: "POST",
        body: buildFormData({
            audioBlob,
            transcript,
            assignmentId,
            itemId,
        }),
    });

    if (!response.ok) {
        throw new Error("Failed to submit pronunciation audio.");
    }

    return response.json();
};

export const getPronunciationJobStatus = async (jobId) => {
    const response = await fetch(
        `${API_BASE_URL}/api/pronunciation/jobs/${jobId}`
    );

    if (!response.ok) {
        throw new Error("Failed to fetch pronunciation job status.");
    }

    const data = await response.json();

    return {
        ...data,
        result: data.result ? normalizeResult(data.result) : null,
    };
};