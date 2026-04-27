import { ASSIGNMENT_STATUS, mockAssignments } from "./mockAssignments";

const ASSIGNMENTS_STORAGE_KEY = "pronunciation_assignments";

const isBrowser = () => {
    return typeof window !== "undefined" && Boolean(window.localStorage);
};

const cloneData = (data) => {
    return JSON.parse(JSON.stringify(data));
};

const saveAssignmentsToStorage = (assignments) => {
    if (!isBrowser()) return;

    window.localStorage.setItem(
        ASSIGNMENTS_STORAGE_KEY,
        JSON.stringify(assignments)
    );
};

const getInitialAssignments = () => {
    return cloneData(mockAssignments);
};

const loadAssignmentsFromStorage = () => {
    if (!isBrowser()) {
        return getInitialAssignments();
    }

    const storedAssignments = window.localStorage.getItem(ASSIGNMENTS_STORAGE_KEY);

    if (!storedAssignments) {
        const initialAssignments = getInitialAssignments();
        saveAssignmentsToStorage(initialAssignments);
        return initialAssignments;
    }

    try {
        const parsedAssignments = JSON.parse(storedAssignments);

        if (!Array.isArray(parsedAssignments)) {
            const initialAssignments = getInitialAssignments();
            saveAssignmentsToStorage(initialAssignments);
            return initialAssignments;
        }

        return parsedAssignments;
    } catch (error) {
        console.error("Failed to parse assignments from localStorage:", error);

        const initialAssignments = getInitialAssignments();
        saveAssignmentsToStorage(initialAssignments);
        return initialAssignments;
    }
};

const calculateProgress = (items = []) => {
    const totalItems = items.length;
    const completedItems = items.filter((item) => item.completed).length;

    const scoredItems = items.filter(
        (item) => typeof item.latestScore === "number"
    );

    const averageScore =
        scoredItems.length > 0
            ? Math.round(
                scoredItems.reduce((total, item) => total + item.latestScore, 0) /
                scoredItems.length
            )
            : null;

    const percent =
        totalItems > 0 ? Math.round((completedItems / totalItems) * 100) : 0;

    return {
        completedItems,
        totalItems,
        percent,
        averageScore,
    };
};

const resolveAssignmentStatus = (assignment) => {
    const now = new Date();
    const deadline = new Date(assignment.deadline);

    if (assignment.progress?.percent === 100) {
        return ASSIGNMENT_STATUS.COMPLETED;
    }

    if (deadline < now) {
        return ASSIGNMENT_STATUS.LATE;
    }

    if (assignment.isNew) {
        return ASSIGNMENT_STATUS.NEW;
    }

    return ASSIGNMENT_STATUS.IN_PROGRESS;
};

export const initializeAssignments = () => {
    const assignments = loadAssignmentsFromStorage();
    saveAssignmentsToStorage(assignments);
    return assignments;
};

export const getAssignments = () => {
    return loadAssignmentsFromStorage();
};

export const getAssignmentById = (assignmentId) => {
    const assignments = loadAssignmentsFromStorage();

    return assignments.find((assignment) => assignment.id === assignmentId) || null;
};

export const createAssignment = (assignmentData) => {
    const assignments = loadAssignmentsFromStorage();

    const items = assignmentData.items || [];

    const newAssignment = {
        id: assignmentData.id || `assignment-${Date.now()}`,
        title: assignmentData.title,
        description: assignmentData.description || "",
        teacherName: assignmentData.teacherName || "Teacher",
        studentName: assignmentData.studentName || "Student",
        createdAt: assignmentData.createdAt || new Date().toISOString(),
        deadline: assignmentData.deadline,
        priority: assignmentData.priority || "medium",
        status: assignmentData.status || ASSIGNMENT_STATUS.NEW,
        isNew: assignmentData.isNew ?? true,
        items,
        progress: assignmentData.progress || calculateProgress(items),
        results: assignmentData.results || [],
        teacherFeedback: assignmentData.teacherFeedback || "",
    };

    const updatedAssignments = [newAssignment, ...assignments];
    saveAssignmentsToStorage(updatedAssignments);

    return newAssignment;
};

export const updateAssignment = (assignmentId, updates) => {
    const assignments = loadAssignmentsFromStorage();

    let updatedAssignment = null;

    const updatedAssignments = assignments.map((assignment) => {
        if (assignment.id !== assignmentId) {
            return assignment;
        }

        updatedAssignment = {
            ...assignment,
            ...updates,
            id: assignment.id,
        };

        return updatedAssignment;
    });

    saveAssignmentsToStorage(updatedAssignments);

    return updatedAssignment;
};

export const updateAssignmentStatus = (assignmentId, status) => {
    return updateAssignment(assignmentId, {
        status,
        isNew: status === ASSIGNMENT_STATUS.NEW,
    });
};

export const markAssignmentAsViewed = (assignmentId) => {
    const assignment = getAssignmentById(assignmentId);

    if (!assignment) return null;

    const nextStatus =
        assignment.status === ASSIGNMENT_STATUS.NEW
            ? ASSIGNMENT_STATUS.IN_PROGRESS
            : assignment.status;

    return updateAssignment(assignmentId, {
        isNew: false,
        status: nextStatus,
    });
};

export const updateAssignmentItemResult = ({
    assignmentId,
    itemId,
    score,
    audioUrl = null,
    phonemeErrors = [],
}) => {
    const assignment = getAssignmentById(assignmentId);

    if (!assignment) return null;

    const updatedItems = assignment.items.map((item) => {
        if (item.id !== itemId) {
            return item;
        }

        return {
            ...item,
            completed: true,
            latestScore: score,
        };
    });

    const updatedResult = {
        id: `result-${Date.now()}`,
        itemId,
        score,
        submittedAt: new Date().toISOString(),
        audioUrl,
        phonemeErrors,
    };

    const updatedAssignmentDraft = {
        ...assignment,
        isNew: false,
        items: updatedItems,
        results: [updatedResult, ...assignment.results],
        progress: calculateProgress(updatedItems),
    };

    const updatedAssignment = {
        ...updatedAssignmentDraft,
        status: resolveAssignmentStatus(updatedAssignmentDraft),
    };

    return updateAssignment(assignmentId, updatedAssignment);
};

export const resetAssignmentsStorage = () => {
    const initialAssignments = getInitialAssignments();
    saveAssignmentsToStorage(initialAssignments);
    return initialAssignments;
};

export const clearAssignmentsStorage = () => {
    if (!isBrowser()) return;

    window.localStorage.removeItem(ASSIGNMENTS_STORAGE_KEY);
};