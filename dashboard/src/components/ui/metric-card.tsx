interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string;
  accentColor?: string;
}

export function MetricCard({ label, value, delta, accentColor = "#6366F1" }: MetricCardProps) {
  return (
    <div
      className="bg-white rounded-2xl px-5 py-5 border-2 border-[#E0E7FF] transition-all hover:-translate-y-0.5 hover:shadow-lg"
      style={{
        borderTop: `5px solid ${accentColor}`,
        boxShadow: "0 4px 18px rgba(79,70,229,0.10)",
      }}
    >
      <p className="text-[11px] font-extrabold uppercase tracking-[0.1em] text-[#7C3AED] mb-2">
        {label}
      </p>
      <p
        className="text-[28px] font-bold text-[#1E1B4B] tracking-tight leading-none"
        style={{ fontFamily: "'JetBrains Mono', monospace" }}
      >
        {value}
      </p>
      {delta && (
        <p className="text-[13px] font-semibold text-[#6B7280] mt-1.5">{delta}</p>
      )}
    </div>
  );
}
