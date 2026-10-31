import { useEffect, useState } from 'react';
import { Grid } from './pages/Grid';
import { GazeCheck } from './pages/GazeCheck';
import { MicCheck } from './pages/MicCheck';
import { TemplateCheck } from './pages/TemplateCheck';

const PAGES = [
  { path: '', title: 'Grid', Page: Grid },
  { path: 'gaze', title: 'Check 1: eye tracking', Page: GazeCheck },
  { path: 'mic', title: 'Check 2: microphone', Page: MicCheck },
  { path: 'templates', title: 'Check 3: voice templates', Page: TemplateCheck },
];

function currentPath(): string {
  return window.location.hash.replace(/^#\/?/, '');
}

export function App() {
  const [path, setPath] = useState(currentPath);

  useEffect(() => {
    const onChange = () => setPath(currentPath());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  const page = PAGES.find((p) => p.path === path) ?? PAGES[0];

  return (
    <div className="app">
      <nav className="nav">
        <strong className="brand">Look-and-Say</strong>
        {PAGES.map((p) => (
          <a key={p.path} href={`#/${p.path}`} className={p === page ? 'active' : undefined}>
            {p.title}
          </a>
        ))}
      </nav>
      <main>
        <page.Page />
      </main>
    </div>
  );
}
