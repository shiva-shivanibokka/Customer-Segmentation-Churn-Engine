export function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2
      className="text-[15px] font-bold text-white rounded-xl px-4 py-2.5 mb-4"
      style={{
        background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)",
        boxShadow: "0 4px 16px rgba(67,56,202,0.28)",
      }}
    >
      {children}
    </h2>
  );
}

export function PageTitle({ children }: { children: React.ReactNode }) {
  return (
    <h1 className="text-[28px] font-extrabold text-[#1E1B4B] tracking-tight pb-3 mb-5 border-b-4 border-[#4F46E5]">
      {children}
    </h1>
  );
}
