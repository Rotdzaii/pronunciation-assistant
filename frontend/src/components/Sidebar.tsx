import React from 'react';
import { 
  Home, 
  Dumbbell as ExerciseIcon, 
  AlertCircle as MistakesIcon, 
  History, 
  User as ProfileIcon 
} from 'lucide-react';
import { cn } from '../lib/utils';
import { Screen } from '../types';

interface SidebarProps {
  active: Screen;
  onChange: (s: Screen) => void;
}

export const Sidebar = ({ active, onChange }: SidebarProps) => {
  const items = [
    { id: 'DASHBOARD', label: 'Trang chủ', icon: Home },
    { id: 'PRACTICE_MENU', label: 'Luyện tập', icon: ExerciseIcon },
    { id: 'MISTAKES', label: 'Lỗi sai', icon: MistakesIcon },
    { id: 'ANALYTICS', label: 'Lịch sử', icon: History },
    { id: 'PROFILE', label: 'Hồ sơ', icon: ProfileIcon },
  ];

  return (
    <nav className="hidden md:flex flex-col h-full sticky left-0 top-0 py-8 bg-surface-container-low border-r border-outline-variant w-64 shrink-0 transition-all">
      <div className="px-6 mb-10">
        <h1 className="text-xl font-bold text-primary tracking-tight">Trợ lý Phát âm</h1>
        <p className="text-xs text-on-surface-variant font-medium mt-1">Học cùng AI</p>
      </div>
      <div className="flex flex-col gap-2 flex-1">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onChange(item.id as Screen)}
            className={cn(
              "flex items-center gap-4 px-6 py-3 mx-2 rounded-xl transition-all font-medium text-sm",
              active === item.id 
                ? "bg-primary text-on-primary shadow-md" 
                : "text-on-surface-variant hover:bg-surface-container-high"
            )}
          >
            <item.icon size={20} fill={active === item.id ? "currentColor" : "none"} />
            {item.label}
          </button>
        ))}
      </div>
    </nav>
  );
};
