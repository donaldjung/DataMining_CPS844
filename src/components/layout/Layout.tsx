import type { ReactNode } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { useEffect, useRef } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';

interface LayoutProps {
  children?: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (mainRef.current) {
      mainRef.current.scrollTo(0, 0);
    }
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen" style={{ background: 'linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col ml-72">
        <Header />
        <main 
          ref={mainRef}
          className="flex-1 p-8 overflow-y-auto"
          style={{ maxHeight: 'calc(100vh - 64px)' }}
        >
          {children || <Outlet />}
        </main>
      </div>
    </div>
  );
}
