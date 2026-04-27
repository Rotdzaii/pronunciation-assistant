import { Link } from "react-router-dom";
import { ASSIGNMENT_STATUS } from "../mockAssignments";

const statusStyle = {
    [ASSIGNMENT_STATUS.NEW]: "bg-blue-100 text-blue-700",
    [ASSIGNMENT_STATUS.IN_PROGRESS]: "bg-yellow-100 text-yellow-700",
    [ASSIGNMENT_STATUS.COMPLETED]: "bg-green-100 text-green-700",
    [ASSIGNMENT_STATUS.LATE]: "bg-red-100 text-red-700",
};

export default function AssignmentCard({ assignment }) {
    return (
        <Link
            to={`/assignments/${assignment.id}`}
            className="block rounded-3xl bg-white p-5 shadow-sm hover:shadow-lg transition"
        >
            <div className="flex justify-between items-center mb-3">
                {assignment.isNew && (
                    <span className="bg-blue-600 text-white px-3 py-1 text-xs rounded-full font-bold">
                        NEW
                    </span>
                )}

                <span
                    className={`px-3 py-1 rounded-full text-xs font-bold ${statusStyle[assignment.status]
                        }`}
                >
                    {assignment.status}
                </span>
            </div>

            <h3 className="text-lg font-extrabold">{assignment.title}</h3>

            <p className="text-sm text-slate-500 mt-1">
                {assignment.description}
            </p>

            <div className="mt-4 text-sm">
                <p>
                    <b>Teacher:</b> {assignment.teacherName}
                </p>
                <p>
                    <b>Deadline:</b>{" "}
                    {new Date(assignment.deadline).toLocaleDateString()}
                </p>
            </div>

            <div className="mt-4">
                <div className="h-2 bg-slate-100 rounded-full">
                    <div
                        className="h-full bg-purple-500 rounded-full"
                        style={{ width: `${assignment.progress.percent}%` }}
                    />
                </div>

                <p className="text-xs mt-1 text-right">
                    {assignment.progress.percent}%
                </p>
            </div>
        </Link>
    );
}