import { useParams, useNavigate } from "react-router-dom";
import { useAssignmentDetail } from "../features/assignments/hooks/useAssignmentDetail";

export default function AssignmentDetailPage() {
    const { assignmentId } = useParams();
    const navigate = useNavigate();
    const { assignment } = useAssignmentDetail(assignmentId);

    if (!assignment) {
        return (
            <main className="min-h-screen flex items-center justify-center">
                Loading...
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8">
            <div className="mx-auto max-w-5xl space-y-6">

                <button
                    onClick={() => navigate("/assignments")}
                    className="text-sm text-purple-600 font-bold"
                >
                    ← Back to Assignments
                </button>

                <div className="bg-white p-6 rounded-3xl shadow-sm">
                    <h1 className="text-3xl font-extrabold">
                        {assignment.title}
                    </h1>

                    <p className="mt-2 text-slate-500">
                        {assignment.description}
                    </p>

                    <div className="mt-4 flex gap-6 text-sm">
                        <p><b>Teacher:</b> {assignment.teacherName}</p>
                        <p><b>Deadline:</b> {new Date(assignment.deadline).toLocaleDateString()}</p>
                        <p><b>Priority:</b> {assignment.priority}</p>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-3xl shadow-sm">
                    <h2 className="text-xl font-extrabold mb-4">
                        Progress
                    </h2>

                    <div className="h-3 bg-slate-100 rounded-full">
                        <div
                            className="h-full bg-purple-500 rounded-full"
                            style={{ width: `${assignment.progress.percent}%` }}
                        />
                    </div>

                    <p className="mt-2 text-sm">
                        {assignment.progress.completedItems}/
                        {assignment.progress.totalItems} items completed
                    </p>
                </div>

                <div className="bg-white p-6 rounded-3xl shadow-sm">
                    <h2 className="text-xl font-extrabold mb-4">
                        Practice Items
                    </h2>

                    <div className="space-y-4">
                        {assignment.items.map((item) => (
                            <div
                                key={item.id}
                                className="p-4 rounded-2xl border border-slate-200"
                            >
                                <p className="font-bold text-lg">
                                    {item.text}
                                </p>

                                {item.note && (
                                    <p className="text-sm text-slate-500 mt-1">
                                        {item.note}
                                    </p>
                                )}

                                <div className="mt-2 text-sm">
                                    <span>
                                        Target: {item.targetPhoneme || "N/A"}
                                    </span>
                                </div>

                                {item.completed && (
                                    <p className="text-green-600 font-bold mt-2">
                                        Completed ({item.latestScore})
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </main>
    );
}