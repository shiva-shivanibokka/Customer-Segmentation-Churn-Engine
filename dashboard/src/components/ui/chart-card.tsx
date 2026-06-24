interface ChartCardProps {
  children: React.ReactNode;
  caption?: string;
}

export function ChartCard({ children, caption }: ChartCardProps) {
  return (
    <div
      className="bg-white rounded-2xl border-2 border-[#E0E7FF] overflow-hidden"
      style={{ boxShadow: "0 4px 18px rgba(79,70,229,0.08)" }}
    >
      <div className="p-4">{children}</div>
      {caption && (
        <div className="px-5 pb-4 text-[13px] text-[#7C3AED] italic">{caption}</div>
      )}
    </div>
  );
}
