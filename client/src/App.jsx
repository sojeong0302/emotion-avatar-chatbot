import { BrowserRouter, Routes, Route } from "react-router-dom";
import ChattingPage from "./pages/ChattingPage/ChattingPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChattingPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
