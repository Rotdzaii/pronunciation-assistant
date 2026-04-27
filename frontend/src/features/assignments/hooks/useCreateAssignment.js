import { useMemo, useState } from "react";
import { createAssignment } from "../assignmentStorage";
import { ASSIGNMENT_PRIORITY } from "../mockAssignments";

const mockDetectedErrors = [
    {
        id: "error-001",
        type: "word",
        text: "think",
        transcript: "think",
        targetPhoneme: "θ",
        errorType: "substitution",
        previousScore: 58,
        note: "Sinh viên phát âm /θ/ giống /t/. Cần luyện vị trí lưỡi.",
    },
    {
        id: "error-002",
        type: "word",
        text: "thank",
        transcript: "thank",
        targetPhoneme: "θ",
        errorType: "substitution",
        previousScore: 61,
        note: "Âm đầu /θ/ chưa rõ, hơi bị bật thành /t/.",
    },
    {
        id: "error-003",
        type: "word",
        text: "ship",
        transcript: "ship",
        targetPhoneme: "ʃ",
        errorType: "contrast_error",
        previousScore: 64,
        note: "Dễ nhầm /ʃ/ với /s/. Cần làm tròn môi nhẹ.",
    },
    {
        id: "error-004",
        type: "sentence",
        text: "She sells fresh fish.",
        transcript: "She sells fresh fish.",
        targetPhoneme: "ʃ",
        errorType: "contrast_error",
        previousScore: 66,
        note: "Cần phân biệt rõ she /ʃiː/ và sells /selz/.",
    },
    {
        id: "error-005",
        type: "sentence",
        text: "Can you help me with this?",
        transcript: "Can you help me with this?",
        targetPhoneme: null,
        errorType: "sentence_stress",
        previousScore: 69,
        note: "Nhịp câu còn đều đều, cần nhấn help và this.",
    },
];

const initialForm = {
    title: "",
    description: "",
    deadline: "",
    priority: ASSIGNMENT_PRIORITY.MEDIUM,
};

export const useCreateAssignment = () => {
    const [form, setForm] = useState(initialForm);
    const [selectedErrorIds, setSelectedErrorIds] = useState([]);
    const [createdAssignment, setCreatedAssignment] = useState(null);
    const [errorMessage, setErrorMessage] = useState("");

    const selectedErrors = useMemo(() => {
        return mockDetectedErrors.filter((item) =>
            selectedErrorIds.includes(item.id)
        );
    }, [selectedErrorIds]);

    const canSubmit =
        form.title.trim() &&
        form.description.trim() &&
        form.deadline &&
        selectedErrors.length > 0;

    const updateField = (field, value) => {
        setForm((current) => ({
            ...current,
            [field]: value,
        }));

        setErrorMessage("");
    };

    const toggleError = (errorId) => {
        setSelectedErrorIds((current) => {
            if (current.includes(errorId)) {
                return current.filter((id) => id !== errorId);
            }

            return [...current, errorId];
        });

        setErrorMessage("");
    };

    const submitAssignment = () => {
        if (!canSubmit) {
            setErrorMessage(
                "Vui lòng nhập đầy đủ title, description, deadline và chọn ít nhất 1 lỗi phát âm."
            );
            return null;
        }

        const assignmentItems = selectedErrors.map((item, index) => ({
            id: `teacher-item-${Date.now()}-${index}`,
            type: item.type,
            text: item.text,
            transcript: item.transcript,
            targetPhoneme: item.targetPhoneme,
            errorType: item.errorType,
            previousScore: item.previousScore,
            completed: false,
            latestScore: null,
            note: item.note,
        }));

        const newAssignment = createAssignment({
            title: form.title.trim(),
            description: form.description.trim(),
            teacherName: "Teacher Demo",
            studentName: "Nguyen Van A",
            deadline: new Date(form.deadline).toISOString(),
            priority: form.priority,
            isNew: true,
            items: assignmentItems,
            results: [],
            teacherFeedback: "",
        });

        setCreatedAssignment(newAssignment);

        return newAssignment;
    };

    return {
        form,
        detectedErrors: mockDetectedErrors,
        selectedErrorIds,
        selectedErrors,
        createdAssignment,
        errorMessage,
        canSubmit,
        updateField,
        toggleError,
        submitAssignment,
    };
};