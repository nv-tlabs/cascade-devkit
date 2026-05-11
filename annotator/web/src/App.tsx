// Phase 1 placeholder. Real wiring (store → Sidebar → VideoPlayer →
// Timeline → RightPanel + LockBar) lands in Phase 2.
export default function App() {
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-[#0c0c1a] text-[#e0e0e0]">
      <div className="text-center">
        <h1 className="text-2xl font-semibold">causal-av-annotator</h1>
        <p className="mt-2 text-sm text-[#8888aa]">
          Phase 1 scaffold. UI wiring coming in Phase 2.
        </p>
        <p className="mt-1 text-xs text-[#8888aa]">
          The API is live at <code>/api/health</code>.
        </p>
      </div>
    </div>
  )
}
