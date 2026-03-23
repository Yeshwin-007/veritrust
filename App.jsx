import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import AnalyzePage from './pages/AnalyzePage';
import ResultPage  from './pages/ResultPage';
import Navbar      from './components/Navbar';

export default function App() {
  return (
    <Router>
      <div className='min-h-screen bg-slate-50'>
        <Navbar />
        <Routes>
          <Route path='/'           element={<LandingPage />} />
          <Route path='/analyze'    element={<AnalyzePage />} />
          <Route path='/result/:id' element={<ResultPage />} />
        </Routes>
      </div>
    </Router>
  );
}