import { useNavigate } from "react-router-dom";
import AppLayout from "../layouts/AppLayout";
import SelectableErrorItem from "../features/assignments/components/SelectableErrorItem";
import { useCreateAssignment } from "../features/assignments/hooks/useCreateAssignment";
import { ASSIGNMENT_PRIORITY } from "../features/assignments/mockAssignments";

export default function TeacherCreateAssignmentPage() {
    const navigate = useNavigate();

    const {
        form,
        detectedErrors,
        selectedErrorIds,
        selectedErrors,
        errorMessage,
        canSubmit,
        updateField,
        toggleError,
        submitAssignment,
    } = useCreateAssignment();

    function handleSubmit(event) {
        event.preventDefault();

        const assignment = submitAssignment();

        if (assignment) {
            navigate("/assignments");
        }
    }

    return (
        <AppLayout>
            <div className="mx-auto max-w-7xl space-y-6">
                <header className="rounded-3xl bg-purple-600 p-8 text-white shadow-sm">
                    <p className="text-sm font-extrabold uppercase text-purple-100">
                        Teacher Assignment Builder
                    </p>

                    <h1 className="mt-2 text-4xl font-extrabold">
                        Tạo bài luyện phát âm lại
                    </h1>

                    <p className="mt-3 max-w-3xl text-purple-100">
                        Chọn các từ hoặc âm vị sinh viên phát âm sai, sau đó tạo assignment
                        luyện lại với deadline và mức độ ưu tiên.
                    </p>
                </header>

                <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-[1fr_420px]">
                    <section className="space-y-4">
                        <div className="rounded-3xl bg-white p-6 shadow-sm">
                            <div className="mb-5 flex items-center justify-between gap-4">
                                <div>
                                    <h2 className="text-2xl font-extrabold text-slate-900">
                                        Detected pronunciation errors
                                    </h2>

                                    <p className="mt-1 text-sm text-slate-500">
                                        Mock data từ kết quả AI. Teacher chọn lỗi để giao bài luyện.
                                    </p>
                                </div>

                                <div className="rounded-2xl bg-purple-50 px-4 py-3 text-center">
                                    <p className="text-xs font-bold uppercase text-purple-400">
                                        Selected
                                    </p>
                                    <p className="text-2xl font-extrabold text-purple-600">
                                        {selectedErrors.length}
                                    </p>
                                </div>
                            </div>

                            <div className="space-y-3">
                                {detectedErrors.map((item) => (
                                    <SelectableErrorItem
                                        key={item.id}
                                        item={item}
                                        selected={selectedErrorIds.includes(item.id)}
                                        onToggle={toggleError}
                                    />
                                ))}
                            </div>
                        </div>
                    </section>

                    <aside className="space-y-4">
                        <div className="rounded-3xl bg-white p-6 shadow-sm">
                            <h2 className="text-2xl font-extrabold text-slate-900">
                                Assignment setup
                            </h2>

                            <div className="mt-5 space-y-4">
                                <label className="block">
                                    <span className="text-sm font-bold text-slate-600">
                                        Title
                                    </span>
                                    <input
                                        value={form.title}
                                        onChange={(event) =>
                                            updateField("title", event.target.value)
                                        }
                                        placeholder="VD: Luyện lại âm /θ/"
                                        className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-purple-400"
                                    />
                                </label>

                                <label className="block">
                                    <span className="text-sm font-bold text-slate-600">
                                        Description
                                    </span>
                                    <textarea
                                        value={form.description}
                                        onChange={(event) =>
                                            updateField("description", event.target.value)
                                        }
                                        placeholder="Nhập yêu cầu cụ thể cho sinh viên..."
                                        rows={5}
                                        className="mt-2 w-full resize-none rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-purple-400"
                                    />
                                </label>

                                <label className="block">
                                    <span className="text-sm font-bold text-slate-600">
                                        Deadline
                                    </span>
                                    <input
                                        type="date"
                                        value={form.deadline}
                                        onChange={(event) =>
                                            updateField("deadline", event.target.value)
                                        }
                                        className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-purple-400"
                                    />
                                </label>

                                <label className="block">
                                    <span className="text-sm font-bold text-slate-600">
                                        Priority
                                    </span>
                                    <select
                                        value={form.priority}
                                        onChange={(event) =>
                                            updateField("priority", event.target.value)
                                        }
                                        className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm outline-none focus:border-purple-400"
                                    >
                                        <option value={ASSIGNMENT_PRIORITY.LOW}>Low</option>
                                        <option value={ASSIGNMENT_PRIORITY.MEDIUM}>Medium</option>
                                        <option value={ASSIGNMENT_PRIORITY.HIGH}>High</option>
                                    </select>
                                </label>
                            </div>
                        </div>

                        <div className="rounded-3xl bg-white p-6 shadow-sm">
                            <h3 className="text-xl font-extrabold text-slate-900">
                                Preview
                            </h3>

                            {selectedErrors.length === 0 ? (
                                <p className="mt-3 text-sm text-slate-500">
                                    Chưa chọn lỗi phát âm nào.
                                </p>
                            ) : (
                                <div className="mt-4 space-y-3">
                                    {selectedErrors.map((item) => (
                                        <div
                                            key={item.id}
                                            className="rounded-2xl bg-purple-50 p-4"
                                        >
                                            <p className="font-extrabold text-purple-700">
                                                {item.text}
                                            </p>
                                            <p className="mt-1 text-xs font-bold text-purple-500">
                                                Target: {item.targetPhoneme || "Sentence focus"} ·
                                                Score: {item.previousScore}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {errorMessage && (
                                <p className="mt-4 rounded-2xl bg-red-50 p-3 text-sm font-bold text-red-600">
                                    {errorMessage}
                                </p>
                            )}

                            <button
                                type="submit"
                                disabled={!canSubmit}
                                className={`mt-5 w-full rounded-2xl px-5 py-4 font-extrabold text-white transition ${canSubmit
                                        ? "bg-purple-600 hover:bg-purple-700"
                                        : "cursor-not-allowed bg-slate-300"
                                    }`}
                            >
                                Create Assignment
                            </button>
                        </div>
                    </aside>
                </form>
            </div>
        </AppLayout>
    );
}