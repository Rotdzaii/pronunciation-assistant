import AssignmentCard from "../features/assignments/components/AssignmentCard";
import { useAssignments } from "../features/assignments/hooks/useAssignments";

export default function AssignmentsPage() {
    const { assignments } = useAssignments();

    return (
        <main className="min-h-screen bg-[#f7f1ff] p-8">
            <div className="mx-auto max-w-6xl">
                <h1 className="text-3xl font-extrabold mb-6">
                    Assignments
                </h1>

                {assignments.length === 0 ? (
                    <p className="text-slate-500">
                        No assignments yet.
                    </p>
                ) : (
                    <div className="grid grid-cols-3 gap-5">
                        {assignments.map((a) => (
                            <AssignmentCard key={a.id} assignment={a} />
                        ))}
                    </div>
                )}
            </div>
        </main>
    );
}