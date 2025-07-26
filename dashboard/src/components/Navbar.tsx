import React from 'react';
import {
  Menu,
  Bell,
  TrendingUp,
} from 'lucide-react';

interface NavbarProps {
  onMenuClick: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ onMenuClick }) => {
  const [botStatus] = React.useState<'running' | 'stopped' | 'error'>('running');
  const [notifications] = React.useState(3);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center">
          <button
            onClick={onMenuClick}
            className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-slate-700 transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          
          <div className="flex items-center ml-4">
            <TrendingUp className="w-6 h-6 text-emerald-400 mr-2" />
            <h1 className="text-xl font-bold text-white">
              Crypto Trading Bot
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${
            botStatus === 'running' 
              ? 'bg-green-900 text-green-300 border-green-600' 
              : botStatus === 'error' 
              ? 'bg-red-900 text-red-300 border-red-600' 
              : 'bg-gray-900 text-gray-300 border-gray-600'
          }`}>
            {botStatus.toUpperCase()}
          </span>
          
          <button className="relative p-2 rounded-lg text-gray-400 hover:text-white hover:bg-slate-700 transition-colors">
            <Bell className="w-5 h-5" />
            {notifications > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                {notifications}
              </span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navbar;