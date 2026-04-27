import { Navigate, Route, Routes } from "react-router-dom";
import PracticePage from "./pages/PracticePage";
import ResultPage from "./pages/ResultPage";
import HistoryPage from "./pages/HistoryPage";
import ProgressPage from "./pages/ProgressPage";
import AssignmentsPage from "./pages/AssignmentsPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/practice" replace />} />
      <Route path="/practice" element={<PracticePage />} />
      <Route path="/result" element={<ResultPage />} />
      <Route path="/history" element={<HistoryPage />} />
      <Route path="/progress" element={<ProgressPage />} />
      <Route path="/assignments" element={<AssignmentsPage />} />
    </Routes>
  );
}