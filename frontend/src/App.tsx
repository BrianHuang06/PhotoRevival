import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "@/components/Layout/Header";
import Workshop from "@/pages/Workshop";
import History from "@/pages/History";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-dark-950 bg-noise font-body">
        <Header />
        <Routes>
          <Route path="/" element={<Workshop />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </div>
    </Router>
  );
}
