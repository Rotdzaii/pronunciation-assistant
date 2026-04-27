import { useParams, useNavigate } from "react-router-dom";
import AppLayout from "../layouts/AppLayout";
import StudentResultCard from "../features/assignments/components/StudentResultCard";
import { useTeacherReview } from "../features/assignments/hooks/useTeacherReview";

export default function TeacherAssignmentReviewPage() {
    const { assignmentId } = useParams();
    const navigate = useNavigate();

    const {
        assignment,
        comment,
        setComment,
        improvement,
        saveComment,
    } = useTeacherReview(assignmentId);

    if (!assignment) {
        return <div className="p-10">Loading...</div>;
    }

    return (
        <AppLayout>
            <div className="mx-auto max-w-6xl space-y-6">

                <button
                    onClick={() => navigate("/teacher/assignments")}
                    className="text-purple-600 font-bold text-sm"
                >
                    ← Back to Dashboard
                </button>

                <header className="bg-purple-600 text-white p-6 rounded-3xl">
                    <h1 className="text-3xl font-extrabold">
                        {assignment.title}
                    </h1>

                    <p className="mt-2 text-purple-100">
                        Student: {assignment.studentName}
                    </p>
                </header>

                <section className="bg-white p-6 rounded-3xl shadow-sm">
                    <h2 className="text-xl font-extrabold mb-4">
                        Improvement Overview
                    </h2>

                    {improvement?.length === 0 ? (
                        <p className="text-slate-500">
                            No completed items yet.
                        </p>
                    ) : (
                        <div className="space-y-3">
                            {improvement.map((item) => (
                                <div
                                    key={item.text}
                                    className="flex justify-between bg-slate-50 p-4 rounded-xl"
                                >
                                    <span className="font-bold">{item.text}</span>

                                    <span className="font-extrabold">
                                        {item.before} → {item.after} (
                                        {item.diff >= 0 ? "+" : ""}
                                        {item.diff})
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </section>

                <section className="bg-white p-6 rounded-3xl shadow-sm">
                    <h2 className="text-xl font-extrabold mb-4">
                        Detailed Results
                    </h2>

                    <div className="space-y-4">
                        {assignment.items.map((item) => (
                            <StudentResultCard key={item.id} item={item} />
                        ))}
                    </div>
                </section>

                <section className="bg-white p-6 rounded-3xl shadow-sm">
                    <h2 className="text-xl font-extrabold mb-4">
                        Teacher Comment
                    </h2>

                    <textarea
                        value={comment}
                        onChange={(e) => setComment(e.target.value)}
                        rows={4}
                        className="w-full border rounded-xl p-4"
                        placeholder="Nhập nhận xét cho sinh viên..."
                    />

                    <button
                        onClick={saveComment}
                        className="mt-4 bg-purple-600 text-white px-6 py-3 rounded-xl font-bold"
                    >
                        Save Comment
                    </button>
                </section>

            </div>
        </AppLayout>
    );
}