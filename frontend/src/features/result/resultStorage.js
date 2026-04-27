const RESULT_STORAGE_KEY = "pronunciation_latest_result";

export function saveLatestResult(result) {
    localStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(result));
}

export function getLatestResult() {
    const rawResult = localStorage.getItem(RESULT_STORAGE_KEY);

    if (!rawResult) return null;

    try {
        return JSON.parse(rawResult);
    } catch {
        return null;
    }
}

export function clearLatestResult() {
    localStorage.removeItem(RESULT_STORAGE_KEY);
}