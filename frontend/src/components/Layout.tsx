import React from 'react';
import { User } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { BottomNav } from './BottomNav';
import { Screen } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  active: Screen;
  onChange: (s: Screen) => void;
}

export const Layout = ({ children, active, onChange }: LayoutProps) => {
  return (
    <div className="flex min-h-screen w-full bg-background no-scrollbar overflow-x-hidden">
      <Sidebar active={active} onChange={onChange} />
      <div className="flex-1 flex flex-col items-center overflow-y-auto no-scrollbar pb-24 md:pb-8">
        <header className="flex justify-between items-center w-full max-w-4xl px-6 h-16 sticky top-0 z-40 bg-[#faf8ffcc] backdrop-blur-md md:hidden">
            <h1 className="text-lg font-bold text-primary">Trợ lý Phát âm</h1>
            <User className="text-on-surface-variant bg-surface-container rounded-full p-1" size={32} />
        </header>
        <div className="w-full max-w-2xl px-6 pt-6">
          {children}
        </div>
      </div>
      <BottomNav active={active} onChange={onChange} />
    </div>
  );
};
