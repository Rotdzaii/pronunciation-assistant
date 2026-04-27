import { Navigate, Route, Routes } from "react-router-dom";
import PracticePage from "./features/practice/PracticePage";
import ResultPage from "./features/result/ResultPage";
import HistoryPage from "./features/history/HistoryPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/practice" replace />} />
      <Route path="/practice" element={<PracticePage />} />
      <Route path="/result" element={<ResultPage />} />
      <Route path="/history" element={<HistoryPage />} />
    </Routes>
  );
}