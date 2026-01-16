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
    <div style={{ 
      display: 'flex', 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%)' 
    }}>
      <Sidebar />
      <div style={{ 
        flex: 1, 
        display: 'flex', 
        flexDirection: 'column', 
        marginLeft: '288px' 
      }}>
        <Header />
        <main 
          ref={mainRef}
          style={{ 
            flex: 1, 
            padding: '32px', 
            overflowY: 'auto',
            maxHeight: 'calc(100vh - 64px)' 
          }}
        >
          {children || <Outlet />}
        </main>
      </div>
    </div>
  );
}
