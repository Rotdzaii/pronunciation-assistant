import { Link } from "react-router-dom";
import { ASSIGNMENT_STATUS } from "../mockAssignments";

const statusConfig = {
    [ASSIGNMENT_STATUS.NEW]: {
        label: "New",
        className: "bg-blue-100 text-blue-700",
    },
    [ASSIGNMENT_STATUS.IN_PROGRESS]: {
        label: "In progress",
        className: "bg-yellow-100 text-yellow-700",
    },
    [ASSIGNMENT_STATUS.COMPLETED]: {
        label: "Completed",
        className: "bg-green-100 text-green-700",
    },
    [ASSIGNMENT_STATUS.LATE]: {
        label: "Late",
        className: "bg-red-100 text-red-700",
    },
};

export default function TeacherAssignmentRow({ assignment }) {
    const progress = assignment.progress || {
        completedItems: 0,
        totalItems: 0,
        percent: 0,
        averageScore: null,
    };

    const status = statusConfig[assignment.status] || {
        label: assignment.status,
        className: "bg-slate-100 text-slate-600",
    };

    return (
        <div className="grid grid-cols-[1.4fr_1fr_120px_140px_130px] items-center gap-4 rounded-2xl bg-white p-5 shadow-sm">
            <div>
                <p className="text-lg font-extrabold text-slate-900">
                    {assignment.title}
                </p>

                <p className="mt-1 line-clamp-1 text-sm text-slate-500">
                    {assignment.description}
                </p>

                <p className="mt-2 text-xs font-bold text-slate-400">
                    Deadline: {new Date(assignment.deadline).toLocaleDateString("vi-VN")}
                </p>
            </div>

            <div>
                <p className="text-xs font-bold uppercase text-slate-400">Student</p>
                <p className="mt-1 font-extrabold text-slate-700">
                    {assignment.studentName || "Nguyen Van A"}
                </p>

                <p className="mt-1 text-xs text-slate-400">
                    {progress.completedItems}/{progress.totalItems} items submitted
                </p>
            </div>

            <div>
                <p className="text-xs font-bold uppercase text-slate-400">Avg score</p>
                <p className="mt-1 text-2xl font-extrabold text-purple-600">
                    {progress.averageScore !== null ? progress.averageScore : "--"}
                </p>
            </div>

            <div>
                <p className="mb-2 text-xs font-bold uppercase text-slate-400">
                    Progress
                </p>

                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                    <div
                        className="h-full rounded-full bg-purple-500"
                        style={{ width: `${progress.percent}%` }}
                    />
                </div>

                <p className="mt-1 text-right text-xs font-extrabold text-purple-600">
                    {progress.percent}%
                </p>
            </div>

            <div className="flex flex-col items-end gap-3">
                <span
                    className={`rounded-full px-3 py-1 text-xs font-extrabold ${status.className}`}
                >
                    {status.label}
                </span>

                <Link
                    to={`/teacher/assignments/${assignment.id}`}
                    className="rounded-xl bg-purple-600 px-4 py-2 text-sm font-extrabold text-white"
                >
                    Review
                </Link>
            </div>
        </div>
    );
}