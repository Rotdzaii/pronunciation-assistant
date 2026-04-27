import { Navigate, Route, Routes } from "react-router-dom";
import PracticePage from "./pages/PracticePage";
import ResultPage from "./pages/ResultPage";
import HistoryPage from "./pages/HistoryPage";
import ProgressPage from "./pages/ProgressPage";
import AssignmentsPage from "./pages/AssignmentsPage";
import AssignmentDetailPage from "./pages/AssignmentDetailPage";
import AssignmentPracticePage from "./pages/AssignmentPracticePage";
import TeacherCreateAssignmentPage from "./pages/TeacherCreateAssignmentPage";
import TeacherAssignmentDashboardPage from "./pages/TeacherAssignmentDashboardPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/practice" replace />} />
      <Route path="/practice" element={<PracticePage />} />
      <Route path="/result" element={<ResultPage />} />
      <Route path="/history" element={<HistoryPage />} />
      <Route path="/progress" element={<ProgressPage />} />
      <Route path="/assignments" element={<AssignmentsPage />} />
      <Route
        path="/assignments/:assignmentId"
        element={<AssignmentDetailPage />}
      />
      <Route
        path="/assignments/:assignmentId/practice"
        element={<AssignmentPracticePage />}
      />
      <Route
        path="/teacher/assignments/create"
        element={<TeacherCreateAssignmentPage />}
      />
      <Route
        path="/teacher/assignments"
        element={<TeacherAssignmentDashboardPage />}
      />
    </Routes>
  );
}