export function Logo({ light = false, compact = false }: { light?: boolean; compact?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden>
        <rect width="32" height="32" rx="8" fill={light ? "#1f3a5c" : "#0d1b2e"} />
        <path d="M16 5.5l8.5 3.2v6.4c0 5.4-3.6 9.6-8.5 11.4-4.9-1.8-8.5-6-8.5-11.4V8.7L16 5.5z" fill="#2a78d6" />
        <path d="M11.5 17.5l3 3 6-7" stroke="#fff" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {!compact && (
        <span className={`text-[17px] font-semibold tracking-tight ${light ? "text-white" : "text-ink"}`}>
          ChurnGuard <span className="text-brand-400">AI</span>
        </span>
      )}
    </span>
  );
}
