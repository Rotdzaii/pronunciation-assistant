export default function App() {
  return (
    <main className="min-h-screen bg-[#f7f1ff] p-8 text-slate-900">
      <section className="mx-auto max-w-4xl rounded-3xl bg-white p-10 shadow-sm">
        <p className="text-sm font-bold uppercase text-purple-600">
          Feature 0
        </p>

        <h1 className="mt-3 text-4xl font-extrabold">
          Pronunciation Assistant Frontend
        </h1>

        <p className="mt-4 text-slate-500">
          React + Vite frontend is ready. Next feature will be Practice Audio
          Input: record microphone, upload audio, preview, and submit for AI
          analysis.
        </p>

        <button className="mt-8 rounded-2xl bg-purple-600 px-6 py-3 font-bold text-white">
          Start Practice
        </button>
      </section>
    </main>
  );
}