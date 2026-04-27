import AppLayout from "../layouts/AppLayout";
import TeacherAssignmentRow from "../features/assignments/components/TeacherAssignmentRow";
import { useTeacherAssignments } from "../features/assignments/hooks/useTeacherAssignments";

export default function TeacherAssignmentDashboardPage() {
    const { assignments, stats } = useTeacherAssignments();

    return (
        <AppLayout>
            <div className="mx-auto max-w-7xl space-y-6">
                <header className="rounded-3xl bg-purple-600 p-8 text-white shadow-sm">
                    <p className="text-sm font-extrabold uppercase text-purple-100">
                        Teacher Dashboard
                    </p>

                    <h1 className="mt-2 text-4xl font-extrabold">
                        Theo dõi assignment đã giao
                    </h1>

                    <p className="mt-3 max-w-3xl text-purple-100">
                        Giảng viên xem tiến độ nộp bài, điểm trung bình và trạng thái luyện
                        tập của từng sinh viên.
                    </p>
                </header>

                <section className="grid gap-4 md:grid-cols-5">
                    <div className="rounded-3xl bg-white p-5 shadow-sm">
                        <p className="text-xs font-bold uppercase text-slate-400">
                            Total
                        </p>
                        <p className="mt-2 text-3xl font-extrabold text-slate-900">
                            {stats.total}
                        </p>
                    </div>

                    <div className="rounded-3xl bg-white p-5 shadow-sm">
                        <p className="text-xs font-bold uppercase text-slate-400">
                            New
                        </p>
                        <p className="mt-2 text-3xl font-extrabold text-blue-600">
                            {stats.newCount}
                        </p>
                    </div>

                    <div className="rounded-3xl bg-white p-5 shadow-sm">
                        <p className="text-xs font-bold uppercase text-slate-400">
                            In progress
                        </p>
                        <p className="mt-2 text-3xl font-extrabold text-yellow-600">
                            {stats.inProgress}
                        </p>
                    </div>

                    <div className="rounded-3xl bg-white p-5 shadow-sm">
                        <p className="text-xs font-bold uppercase text-slate-400">
                            Completed
                        </p>
                        <p className="mt-2 text-3xl font-extrabold text-green-600">
                            {stats.completed}
                        </p>
                    </div>

                    <div className="rounded-3xl bg-white p-5 shadow-sm">
                        <p className="text-xs font-bold uppercase text-slate-400">
                            Late
                        </p>
                        <p className="mt-2 text-3xl font-extrabold text-red-600">
                            {stats.late}
                        </p>
                    </div>
                </section>

                <section className="rounded-3xl bg-white p-6 shadow-sm">
                    <div className="mb-4 flex items-center justify-between">
                        <div>
                            <h2 className="text-2xl font-extrabold text-slate-900">
                                Class progress overview
                            </h2>

                            <p className="mt-1 text-sm text-slate-500">
                                Average completion across all assigned pronunciation tasks.
                            </p>
                        </div>

                        <div className="rounded-3xl bg-purple-50 px-6 py-4 text-center">
                            <p className="text-xs font-bold uppercase text-purple-400">
                                Avg progress
                            </p>
                            <p className="text-3xl font-extrabold text-purple-600">
                                {stats.averageProgress}%
                            </p>
                        </div>
                    </div>

                    <div className="h-4 overflow-hidden rounded-full bg-slate-100">
                        <div
                            className="h-full rounded-full bg-purple-500"
                            style={{ width: `${stats.averageProgress}%` }}
                        />
                    </div>
                </section>

                <section className="space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-2xl font-extrabold text-slate-900">
                                Assigned practice list
                            </h2>

                            <p className="mt-1 text-sm text-slate-500">
                                Danh sách assignment teacher đã tạo bằng mock storage.
                            </p>
                        </div>
                    </div>

                    {assignments.length === 0 ? (
                        <div className="rounded-3xl bg-white p-10 text-center shadow-sm">
                            <p className="text-xl font-extrabold text-slate-900">
                                Chưa có assignment nào
                            </p>

                            <p className="mt-2 text-sm text-slate-500">
                                Hãy tạo assignment mới từ trang Create Assignment.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {assignments.map((assignment) => (
                                <TeacherAssignmentRow
                                    key={assignment.id}
                                    assignment={assignment}
                                />
                            ))}
                        </div>
                    )}
                </section>
            </div>
        </AppLayout>
    );
}