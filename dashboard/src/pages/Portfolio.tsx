import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';

const Portfolio: React.FC = () => {
  const portfolioData = [
    { name: 'BTC', value: 45, amount: 5625 },
    { name: 'ETH', value: 30, amount: 3750 },
    { name: 'ADA', value: 15, amount: 1875 },
    { name: 'DOT', value: 10, amount: 1250 },
  ];

  const performanceData = [
    { date: '2024-01-01', value: 10000 },
    { date: '2024-01-02', value: 10150 },
    { date: '2024-01-03', value: 10080 },
    { date: '2024-01-04', value: 10320 },
    { date: '2024-01-05', value: 10450 },
    { date: '2024-01-06', value: 10380 },
    { date: '2024-01-07', value: 10620 },
    { date: '2024-01-08', value: 10750 },
    { date: '2024-01-09', value: 10680 },
    { date: '2024-01-10', value: 10890 },
    { date: '2024-01-11', value: 11020 },
    { date: '2024-01-12', value: 10950 },
    { date: '2024-01-13', value: 11180 },
    { date: '2024-01-14', value: 11350 },
    { date: '2024-01-15', value: 11450 },
  ];

  const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6'];

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white mb-6">Portfolio</h1>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
          <p className="text-gray-400 text-sm mb-1">Total Value</p>
          <p className="text-3xl font-bold text-emerald-400">$12,500</p>
          <p className="text-green-400 text-sm mt-1">+14.5% (+$1,500)</p>
        </div>
        
        <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
          <p className="text-gray-400 text-sm mb-1">Available Balance</p>
          <p className="text-3xl font-bold text-blue-400">$2,350</p>
          <p className="text-gray-400 text-sm mt-1">18.8% of portfolio</p>
        </div>
        
        <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
          <p className="text-gray-400 text-sm mb-1">Total P&L</p>
          <p className="text-3xl font-bold text-green-400">+$1,250</p>
          <p className="text-green-400 text-sm mt-1">+11.1% all time</p>
        </div>
        
        <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
          <p className="text-gray-400 text-sm mb-1">Daily P&L</p>
          <p className="text-3xl font-bold text-green-400">+$85</p>
          <p className="text-green-400 text-sm mt-1">+0.68% today</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Allocation */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold text-white mb-4">
            Portfolio Allocation
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={portfolioData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name} ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {portfolioData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '8px',
                    color: '#f1f5f9'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Asset Values */}
        <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold text-white mb-4">
            Asset Values
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={portfolioData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '8px',
                    color: '#f1f5f9'
                  }}
                />
                <Bar dataKey="amount" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Portfolio Performance */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold text-white mb-4">
          Portfolio Performance (15 days)
        </h3>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="date" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#f1f5f9'
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#10b981"
                strokeWidth={3}
                dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Portfolio;