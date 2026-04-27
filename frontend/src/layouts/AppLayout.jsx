import { NavLink } from "react-router-dom";

const navItems = [
    { path: "/practice", label: "Practice", icon: "🎙️" },
    { path: "/result", label: "Result", icon: "📊" },
    { path: "/history", label: "History", icon: "🕘" },
    { path: "/progress", label: "Progress", icon: "📈" },
];

export default function AppLayout({ children }) {
    return (
        <div className="min-h-screen bg-[#f7f1ff] text-slate-900">
            <div className="mx-auto flex min-h-screen max-w-7xl bg-white shadow-sm">
                <aside className="sticky top-0 h-screen w-64 border-r border-purple-100 bg-white p-6">
                    <div className="mb-10">
                        <h1 className="text-2xl font-extrabold text-purple-600">
                            SpeakBetter
                        </h1>
                        <p className="text-xs font-bold uppercase text-slate-400">
                            AI Pronunciation
                        </p>
                    </div>

                    <nav className="space-y-2">
                        {navItems.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                className={({ isActive }) =>
                                    `flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-extrabold transition ${isActive
                                        ? "bg-purple-100 text-purple-700"
                                        : "text-slate-500 hover:bg-slate-50 hover:text-purple-600"
                                    }`
                                }
                            >
                                <span>{item.icon}</span>
                                <span>{item.label}</span>
                            </NavLink>
                        ))}
                    </nav>

                    <div className="mt-10 rounded-3xl bg-purple-50 p-4">
                        <p className="text-sm font-extrabold text-purple-700">
                            Daily Goal
                        </p>
                        <p className="mt-1 text-xs text-purple-500">
                            Practice 10 minutes today to improve your pronunciation.
                        </p>

                        <NavLink
                            to="/practice"
                            className="mt-4 block rounded-2xl bg-purple-600 px-4 py-3 text-center text-sm font-extrabold text-white"
                        >
                            Start Practice
                        </NavLink>
                    </div>
                </aside>

                <main className="flex-1 bg-[#f7f1ff] p-8">{children}</main>
            </div>
        </div>
    );
}