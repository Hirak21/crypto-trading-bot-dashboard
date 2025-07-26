import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  Wallet,
  Brain,
  Settings,
  FileText,
} from 'lucide-react';

interface SidebarProps {
  open: boolean;
  onToggle: () => void;
}

const menuItems = [
  { text: 'Dashboard', icon: LayoutDashboard, path: '/' },
  { text: 'Trading', icon: TrendingUp, path: '/trading' },
  { text: 'Portfolio', icon: Wallet, path: '/portfolio' },
  { text: 'Strategies', icon: Brain, path: '/strategies' },
  { text: 'Settings', icon: Settings, path: '/settings' },
  { text: 'Logs', icon: FileText, path: '/logs' },
];

const Sidebar: React.FC<SidebarProps> = ({ open }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  return (
    <div
      className={`${
        open ? 'w-60' : 'w-16'
      } transition-all duration-300 bg-gradient-to-b from-slate-900 to-slate-800 border-r border-slate-700 flex flex-col`}
    >
      <div className="mt-20 flex-1">
        <nav className="px-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <button
                key={item.text}
                onClick={() => handleNavigation(item.path)}
                className={`w-full flex items-center px-3 py-3 mb-1 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-emerald-600/20 text-emerald-400 border-r-2 border-emerald-400'
                    : 'text-gray-400 hover:bg-slate-700/50 hover:text-white'
                } ${open ? 'justify-start' : 'justify-center'}`}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                {open && (
                  <span className="ml-3 text-sm font-medium">
                    {item.text}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
};

export default Sidebar;