import { Link } from "react-router-dom";
import { ASSIGNMENT_STATUS } from "../mockAssignments";

const statusStyle = {
    [ASSIGNMENT_STATUS.NEW]: "bg-blue-100 text-blue-700",
    [ASSIGNMENT_STATUS.IN_PROGRESS]: "bg-yellow-100 text-yellow-700",
    [ASSIGNMENT_STATUS.COMPLETED]: "bg-green-100 text-green-700",
    [ASSIGNMENT_STATUS.LATE]: "bg-red-100 text-red-700",
};

const statusLabel = {
    [ASSIGNMENT_STATUS.NEW]: "New",
    [ASSIGNMENT_STATUS.IN_PROGRESS]: "In progress",
    [ASSIGNMENT_STATUS.COMPLETED]: "Completed",
    [ASSIGNMENT_STATUS.LATE]: "Late",
};

export default function AssignmentCard({ assignment }) {
    const progress = assignment.progress || {
        completedItems: 0,
        totalItems: 0,
        percent: 0,
        averageScore: null,
    };

    return (
        <Link
            to={`/assignments/${assignment.id}`}
            className="block rounded-3xl bg-white p-5 shadow-sm transition hover:-translate-y-1 hover:shadow-lg"
        >
            <div className="mb-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                    {assignment.isNew && (
                        <span className="rounded-full bg-blue-600 px-3 py-1 text-xs font-bold text-white">
                            NEW
                        </span>
                    )}

                    <span
                        className={`rounded-full px-3 py-1 text-xs font-bold ${statusStyle[assignment.status] || "bg-slate-100 text-slate-600"
                            }`}
                    >
                        {statusLabel[assignment.status] || assignment.status}
                    </span>
                </div>

                <span className="rounded-full bg-purple-50 px-3 py-1 text-xs font-bold text-purple-600">
                    {assignment.priority}
                </span>
            </div>

            <h3 className="text-lg font-extrabold text-slate-900">
                {assignment.title}
            </h3>

            <p className="mt-1 line-clamp-2 text-sm text-slate-500">
                {assignment.description}
            </p>

            <div className="mt-4 space-y-1 text-sm text-slate-600">
                <p>
                    <span className="font-bold">Teacher:</span> {assignment.teacherName}
                </p>

                <p>
                    <span className="font-bold">Deadline:</span>{" "}
                    {new Date(assignment.deadline).toLocaleDateString("vi-VN")}
                </p>
            </div>

            <div className="mt-5">
                <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-500">
                        {progress.completedItems}/{progress.totalItems} items
                    </span>

                    <span className="font-extrabold text-purple-600">
                        {progress.percent}%
                    </span>
                </div>

                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                    <div
                        className="h-full rounded-full bg-purple-500"
                        style={{ width: `${progress.percent}%` }}
                    />
                </div>

                {progress.averageScore !== null && (
                    <p className="mt-2 text-right text-xs font-extrabold text-purple-600">
                        Avg Score: {progress.averageScore}
                    </p>
                )}
            </div>
        </Link>
    );
}