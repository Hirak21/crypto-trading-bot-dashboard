import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Strategies from './pages/Strategies';

// Placeholder components for missing pages
const Trading = () => <div className="p-6"><h1 className="text-2xl font-bold">Trading Page - Coming Soon</h1></div>;
const Settings = () => <div className="p-6"><h1 className="text-2xl font-bold">Settings Page - Coming Soon</h1></div>;
const Logs = () => <div className="p-6"><h1 className="text-2xl font-bold">Logs Page - Coming Soon</h1></div>;

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Router>
      <div className="flex h-screen bg-gray-50">
        <Sidebar open={sidebarOpen} onToggle={handleSidebarToggle} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Navbar onMenuClick={handleSidebarToggle} />
          <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50 p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/trading" element={<Trading />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/strategies" element={<Strategies />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/logs" element={<Logs />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
};

export default App;