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

interface BottomNavProps {
  active: Screen;
  onChange: (s: Screen) => void;
}

export const BottomNav = ({ active, onChange }: BottomNavProps) => {
  const items = [
    { id: 'DASHBOARD', label: 'Trang chủ', icon: Home },
    { id: 'PRACTICE_MENU', label: 'Luyện tập', icon: ExerciseIcon },
    { id: 'MISTAKES', label: 'Lỗi sai', icon: MistakesIcon },
    { id: 'ANALYTICS', label: 'Lịch sử', icon: History },
    { id: 'PROFILE', label: 'Hồ sơ', icon: ProfileIcon },
  ];

  return (
    <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 py-3 bg-white/80 backdrop-blur-xl border-t border-outline-variant/30 shadow-lg rounded-t-2xl">
      {items.map((item) => (
        <button
          key={item.id}
          onClick={() => onChange(item.id as Screen)}
          className={cn(
            "flex flex-col items-center justify-center transition-all p-2 rounded-xl min-w-[64px]",
            active === item.id 
              ? "bg-primary-container text-on-primary-container font-bold" 
              : "text-on-surface-variant"
          )}
        >
          <item.icon size={20} fill={active === item.id ? "currentColor" : "none"} />
          <span className="text-[10px] mt-1">{item.label}</span>
        </button>
      ))}
    </nav>
  );
};
