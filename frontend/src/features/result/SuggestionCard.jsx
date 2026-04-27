export default function SuggestionCard({ suggestion }) {
    return (
        <section className="rounded-3xl border border-yellow-200 bg-yellow-50 p-6">
            <p className="text-sm font-extrabold uppercase text-yellow-700">
                AI Suggestion
            </p>

            <p className="mt-3 text-base font-medium leading-7 text-yellow-800">
                💡 {suggestion}
            </p>
        </section>
    );
}