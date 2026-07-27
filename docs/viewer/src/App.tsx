import { lazy, Suspense, useState } from "react";
import { darkTheme } from "./canvas/theme";

const pages = [
  {
    id: "diff",
    label: "Diff Safetensors",
    Component: lazy(() => import("../../canvases/diff-safetensors.canvas")),
  },
  {
    id: "korznikov",
    label: "Korznikov Dataset",
    Component: lazy(() => import("../../canvases/korznikov-dataset.canvas")),
  },
] as const;

export default function App() {
  const [page, setPage] = useState<(typeof pages)[number]["id"]>("diff");
  const active = pages.find((p) => p.id === page)!;
  const { Component } = active;

  return (
    <div style={{ minHeight: "100vh", background: darkTheme.bg.editor }}>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
          padding: "12px 20px",
          borderBottom: `1px solid ${darkTheme.stroke.primary}`,
          background: darkTheme.bg.chrome,
        }}
      >
        <span style={{ fontWeight: 600, marginRight: 12 }}>NeuralProbes Docs</span>
        {pages.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setPage(p.id)}
            style={{
              border: "none",
              cursor: "pointer",
              padding: "6px 12px",
              borderRadius: 4,
              fontSize: 13,
              background: page === p.id ? darkTheme.accent.control : darkTheme.fill.tertiary,
              color: page === p.id ? darkTheme.text.onAccent : darkTheme.text.secondary,
            }}
          >
            {p.label}
          </button>
        ))}
      </header>
      <main style={{ maxWidth: 900, margin: "0 auto", padding: "16px 20px 48px" }}>
        <Suspense fallback={<p style={{ color: darkTheme.text.secondary }}>Загрузка…</p>}>
          <Component />
        </Suspense>
      </main>
    </div>
  );
}
