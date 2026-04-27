const HISTORY_KEY = "pronunciation_history";

export function getHistory() {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];

    try {
        return JSON.parse(raw);
    } catch {
        return [];
    }
}

export function addToHistory(result) {
    const history = getHistory();

    const newItem = {
        ...result,
        id: result.id || Date.now(),
    };

    const updated = [newItem, ...history];

    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
}