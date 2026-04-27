import { useParams, useNavigate } from "react-router-dom";
import AudioRecorder from "../features/practice/AudioRecorder";
import AudioUploader from "../features/practice/AudioUploader";
import { useAssignmentPractice } from "../features/assignments/hooks/useAssignmentPractice";

export default function AssignmentPracticePage() {
    const { assignmentId } = useParams();
    const navigate = useNavigate();

    const { assignment, currentItem, status, handleAudio, analyze } =
        useAssignmentPractice(assignmentId);

    if (!assignment) {
        return <div className="p-10">Loading...</div>;
    }

    if (!currentItem) {
        return (
            <div className="p-10 text-center">
                <h1 className="text-2xl font-extrabold">
                    🎉 Assignment Completed!
                </h1>

                <button
                    onClick={() => navigate(`/assignments/${assignmentId}`)}
                    className="mt-6 bg-purple-600 text-white px-6 py-3 rounded-xl"
                >
                    Back to Detail
                </button>
            </div>
        );
    }

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8">
            <div className="mx-auto max-w-4xl space-y-6">

                <div className="bg-purple-600 text-white p-4 rounded-2xl">
                    <p className="text-sm">Assignment Mode</p>
                    <h1 className="text-xl font-extrabold">
                        {assignment.title}
                    </h1>
                </div>

                <div className="bg-white p-6 rounded-3xl shadow-sm">
                    <h2 className="text-2xl font-extrabold">
                        {currentItem.text}
                    </h2>

                    {currentItem.note && (
                        <p className="text-slate-500 mt-2">
                            {currentItem.note}
                        </p>
                    )}
                </div>

                <AudioRecorder onAudioReady={handleAudio} />
                <AudioUploader onAudioReady={handleAudio} />

                <button
                    onClick={analyze}
                    className="bg-purple-600 text-white px-6 py-3 rounded-xl font-bold"
                >
                    {status === "processing" ? "Analyzing..." : "Submit"}
                </button>

            </div>
        </main>
    );
}