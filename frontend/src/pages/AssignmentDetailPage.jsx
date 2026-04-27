import { useNavigate, useParams } from "react-router-dom";
import { useAssignmentDetail } from "../features/assignments/hooks/useAssignmentDetail";

const statusLabel = {
    new: "New",
    in_progress: "In progress",
    completed: "Completed",
    late: "Late",
};

const statusStyle = {
    new: "bg-blue-100 text-blue-700",
    in_progress: "bg-yellow-100 text-yellow-700",
    completed: "bg-green-100 text-green-700",
    late: "bg-red-100 text-red-700",
};

export default function AssignmentDetailPage() {
    const { assignmentId } = useParams();
    const navigate = useNavigate();
    const { assignment } = useAssignmentDetail(assignmentId);

    if (!assignment) {
        return (
            <main className="flex min-h-screen items-center justify-center bg-[#f7f1ff] p-8">
                <div className="rounded-3xl bg-white p-8 text-center shadow-sm">
                    <p className="text-lg font-extrabold text-purple-600">
                        Loading assignment...
                    </p>
                </div>
            </main>
        );
    }

    const progress = assignment.progress || {
        completedItems: 0,
        totalItems: 0,
        percent: 0,
        averageScore: null,
    };

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8 text-slate-900">
            <div className="mx-auto max-w-6xl space-y-6">
                <button
                    type="button"
                    onClick={() => navigate("/assignments")}
                    className="text-sm font-extrabold text-purple-600"
                >
                    ← Back to Assignments
                </button>

                <section className="rounded-3xl bg-white p-8 shadow-sm">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                        <div>
                            <p className="text-sm font-extrabold uppercase text-purple-500">
                                Assignment Detail
                            </p>

                            <h1 className="mt-2 text-4xl font-extrabold">
                                {assignment.title}
                            </h1>

                            <p className="mt-3 max-w-3xl text-slate-500">
                                {assignment.description}
                            </p>
                        </div>

                        <div className="flex flex-wrap gap-2">
                            <span
                                className={`rounded-full px-4 py-2 text-sm font-extrabold ${statusStyle[assignment.status] ||
                                    "bg-slate-100 text-slate-600"
                                    }`}
                            >
                                {statusLabel[assignment.status] || assignment.status}
                            </span>

                            <span className="rounded-full bg-purple-50 px-4 py-2 text-sm font-extrabold text-purple-600">
                                Priority: {assignment.priority}
                            </span>
                        </div>
                    </div>

                    <div className="mt-6 grid gap-4 md:grid-cols-3">
                        <div className="rounded-2xl bg-slate-50 p-4">
                            <p className="text-xs font-bold uppercase text-slate-400">
                                Teacher
                            </p>
                            <p className="mt-1 font-extrabold">{assignment.teacherName}</p>
                        </div>

                        <div className="rounded-2xl bg-slate-50 p-4">
                            <p className="text-xs font-bold uppercase text-slate-400">
                                Deadline
                            </p>
                            <p className="mt-1 font-extrabold">
                                {new Date(assignment.deadline).toLocaleDateString("vi-VN")}
                            </p>
                        </div>

                        <div className="rounded-2xl bg-slate-50 p-4">
                            <p className="text-xs font-bold uppercase text-slate-400">
                                Average Score
                            </p>
                            <p className="mt-1 font-extrabold text-purple-600">
                                {progress.averageScore !== null
                                    ? `${progress.averageScore}/100`
                                    : "Not submitted yet"}
                            </p>
                        </div>
                    </div>
                </section>

                <section className="rounded-3xl bg-white p-8 shadow-sm">
                    <div className="mb-4 flex items-center justify-between">
                        <div>
                            <h2 className="text-2xl font-extrabold">Progress Tracking</h2>
                            <p className="mt-1 text-sm text-slate-500">
                                Theo dõi số mục đã luyện, phần trăm hoàn thành và điểm trung
                                bình.
                            </p>
                        </div>

                        <div className="rounded-3xl bg-purple-50 px-6 py-4 text-center text-purple-700">
                            <p className="text-xs font-extrabold uppercase">Completed</p>
                            <p className="text-3xl font-extrabold">{progress.percent}%</p>
                        </div>
                    </div>

                    <div className="h-4 overflow-hidden rounded-full bg-slate-100">
                        <div
                            className="h-full rounded-full bg-purple-500"
                            style={{ width: `${progress.percent}%` }}
                        />
                    </div>

                    <div className="mt-4 grid gap-4 md:grid-cols-3">
                        <div className="rounded-2xl bg-slate-50 p-4">
                            <p className="text-xs font-bold uppercase text-slate-400">
                                Items practiced
                            </p>
                            <p className="mt-1 text-2xl font-extrabold">
                                {progress.completedItems}/{progress.totalItems}
                            </p>
                        </div>

                        <div className="rounded-2xl bg-slate-50 p-4">
                            <p className="text-xs font-bold uppercase text-slate-400">
                                Avg score
                            </p>
                            <p className="mt-1 text-2xl font-extrabold text-purple-600">
                                {progress.averageScore !== null ? progress.averageScore : "--"}
                            </p>
                        </div>

                        <div className="rounded-2xl bg-slate-50 p-4">
                            <p className="text-xs font-bold uppercase text-slate-400">
                                Status
                            </p>
                            <p className="mt-1 text-2xl font-extrabold">
                                {statusLabel[assignment.status] || assignment.status}
                            </p>
                        </div>
                    </div>

                    <button
                        type="button"
                        onClick={() => navigate(`/assignments/${assignment.id}/practice`)}
                        className="mt-6 rounded-2xl bg-purple-600 px-6 py-3 font-extrabold text-white shadow-sm"
                    >
                        Continue Practice
                    </button>
                </section>

                <section className="rounded-3xl bg-white p-8 shadow-sm">
                    <h2 className="text-2xl font-extrabold">Practice Items</h2>

                    <div className="mt-5 space-y-4">
                        {assignment.items.map((item) => (
                            <div
                                key={item.id}
                                className="rounded-2xl border border-slate-200 p-5"
                            >
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div>
                                        <p className="text-xl font-extrabold">{item.text}</p>

                                        {item.note && (
                                            <p className="mt-1 text-sm text-slate-500">{item.note}</p>
                                        )}

                                        <div className="mt-3 flex flex-wrap gap-2 text-xs font-bold">
                                            <span className="rounded-full bg-purple-50 px-3 py-1 text-purple-600">
                                                Type: {item.type}
                                            </span>

                                            <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">
                                                Target: {item.targetPhoneme || "Sentence focus"}
                                            </span>
                                        </div>
                                    </div>

                                    {item.completed ? (
                                        <div className="rounded-2xl bg-green-50 px-4 py-3 text-right">
                                            <p className="text-xs font-bold uppercase text-green-500">
                                                Completed
                                            </p>
                                            <p className="text-2xl font-extrabold text-green-600">
                                                {item.latestScore}
                                            </p>
                                        </div>
                                    ) : (
                                        <div className="rounded-2xl bg-slate-50 px-4 py-3 text-right">
                                            <p className="text-xs font-bold uppercase text-slate-400">
                                                Pending
                                            </p>
                                            <p className="text-sm font-extrabold text-slate-500">
                                                Not practiced
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </section>
            </div>
        </main>
    );
}