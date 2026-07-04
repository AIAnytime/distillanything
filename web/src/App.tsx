import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { api } from "./api";
import { ToastProvider, usePoll } from "./components";
import { JobsDrawer } from "./JobsDrawer";
import { ComparePage } from "./pages/ComparePage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { NewRunPage } from "./pages/NewRunPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunsPage } from "./pages/RunsPage";
import type { Job } from "./types";

function Topbar({ onJobs }: { onJobs: () => void }) {
  const navigate = useNavigate();
  const [activeJobs, setActiveJobs] = useState(0);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("da_theme") ?? "dark",
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("da_theme", theme);
  }, [theme]);

  usePoll(async () => {
    try {
      const jobs = await api<Job[]>("/api/jobs");
      setActiveJobs(jobs.filter((j) => j.status === "running" || j.status === "queued").length);
    } catch {
      /* unauthenticated or offline; badge stays as-is */
    }
  }, 5000);

  return (
    <header className="topbar">
      <NavLink to="/" className="brand">
        <img
          className="brand-logo"
          src={theme === "light" ? "/logo-mark-light.png" : "/logo-mark.png"}
          alt=""
        />
        Distill Anything
      </NavLink>
      <nav className="nav">
        <NavLink to="/" end>
          Runs
        </NavLink>
        <NavLink to="/compare">Compare</NavLink>
        <NavLink to="/datasets">Datasets</NavLink>
      </nav>
      <button className="btn" onClick={onJobs}>
        Jobs{activeJobs > 0 && <span className="badge live num">{activeJobs}</span>}
      </button>
      <button className="btn btn-primary" onClick={() => navigate("/new")}>
        + New run
      </button>
      <button
        className="icon-btn"
        title="Toggle theme"
        aria-label="Toggle theme"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      >
        {theme === "dark" ? "☀" : "☾"}
      </button>
    </header>
  );
}

export default function App() {
  const [jobsOpen, setJobsOpen] = useState(false);
  return (
    <ToastProvider>
      <div className="app">
        <Topbar onJobs={() => setJobsOpen((v) => !v)} />
        <main className="main">
          <Routes>
            <Route path="/" element={<RunsPage />} />
            <Route path="/runs/:name" element={<RunDetailPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/new" element={<NewRunPage />} />
          </Routes>
        </main>
        <JobsDrawer open={jobsOpen} onClose={() => setJobsOpen(false)} />
      </div>
    </ToastProvider>
  );
}
