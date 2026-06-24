"use client";

import { useMemo, useState } from "react";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, AreaChart, Area,
} from "recharts";

const SEGMENT_COLORS: Record<string, string> = {
  "Champions": "#4F46E5",
  "Loyal Customers": "#7C3AED",
  "At-Risk": "#EF4444",
  "Hibernating": "#F59E0B",
  "Lost Customers": "#6B7280",
};
const RISK_COLORS: Record<string, string> = {
  "High Risk": "#EF4444",
  "Medium Risk": "#F59E0B",
  "Low Risk": "#10B981",
};

interface Props { customers: Customer[] }

export function ChurnClient({ customers }: Props) {
  const [selectedSegs, setSelectedSegs] = useState<string[]>([]);

  const allSegments = useMemo(() => [...new Set(customers.map((c) => c.segment))], [customers]);

  const filtered = useMemo(
    () => selectedSegs.length ? customers.filter((c) => selectedSegs.includes(c.segment)) : customers,
    [customers, selectedSegs]
  );

  const kpis = useMemo(() => {
    const total = filtered.length;
    const highRisk = filtered.filter((c) => c.risk_tier === "High Risk").length;
    const avgProb = filtered.reduce((s, c) => s + c.churn_probability, 0) / total;
    const actualChurn = filtered.filter((c) => c.churn === 1).length;
    return { total, highRisk, avgProb, actualChurn };
  }, [filtered]);

  // Histogram bins
  const histData = useMemo(() => {
    const bins = Array.from({ length: 20 }, (_, i) => ({
      range: `${(i * 5).toString().padStart(2, "0")}–${((i + 1) * 5).toString().padStart(2, "0")}%`,
      count: 0,
    }));
    for (const c of filtered) {
      const bin = Math.min(19, Math.floor(c.churn_probability * 20));
      bins[bin].count++;
    }
    return bins;
  }, [filtered]);

  // Risk tier by segment
  const riskBySegData = useMemo(() => {
    const segs = allSegments;
    return segs.map((seg) => {
      const rows = filtered.filter((c) => c.segment === seg);
      return {
        segment: seg,
        "High Risk": rows.filter((c) => c.risk_tier === "High Risk").length,
        "Medium Risk": rows.filter((c) => c.risk_tier === "Medium Risk").length,
        "Low Risk": rows.filter((c) => c.risk_tier === "Low Risk").length,
      };
    });
  }, [filtered, allSegments]);

  // SHAP top features (aggregate)
  const shapData = useMemo(() => {
    const featureMap: Record<string, number[]> = {};
    for (const c of filtered) {
      if (!c.top_shap_features) continue;
      for (const [feat, val] of Object.entries(c.top_shap_features)) {
        if (!featureMap[feat]) featureMap[feat] = [];
        featureMap[feat].push(Math.abs(val));
      }
    }
    return Object.entries(featureMap)
      .map(([feature, vals]) => ({
        feature,
        importance: vals.reduce((a, b) => a + b, 0) / vals.length,
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 12);
  }, [filtered]);

  const toggleSeg = (seg: string) => {
    setSelectedSegs((prev) =>
      prev.includes(seg) ? prev.filter((s) => s !== seg) : [...prev, seg]
    );
  };

  return (
    <div>
      <PageTitle>Churn Risk Dashboard</PageTitle>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Customers" value={kpis.total.toLocaleString()} accentColor="#4F46E5" />
        <MetricCard label="High Risk" value={kpis.highRisk.toLocaleString()} delta={`${((kpis.highRisk / kpis.total) * 100).toFixed(1)}% of total`} accentColor="#EF4444" />
        <MetricCard label="Avg Churn Prob" value={`${(kpis.avgProb * 100).toFixed(1)}%`} accentColor="#F59E0B" />
        <MetricCard label="Actual Churned" value={kpis.actualChurn.toLocaleString()} delta={`${((kpis.actualChurn / kpis.total) * 100).toFixed(1)}% rate`} accentColor="#7C3AED" />
      </div>

      {/* Segment filter */}
      <div className="flex flex-wrap gap-2 mb-6">
        <span className="text-[13px] font-semibold text-[#4F46E5] self-center">Filter:</span>
        {allSegments.map((seg) => (
          <button
            key={seg}
            onClick={() => toggleSeg(seg)}
            className={`px-3 py-1 rounded-full text-[13px] font-semibold border-2 transition-all ${
              selectedSegs.includes(seg)
                ? "text-white border-transparent"
                : "bg-white text-[#6B7280] border-[#DDD6FE] hover:border-[#818CF8]"
            }`}
            style={selectedSegs.includes(seg) ? { background: SEGMENT_COLORS[seg] ?? "#4F46E5" } : {}}
          >
            {seg}
          </button>
        ))}
        {selectedSegs.length > 0 && (
          <button onClick={() => setSelectedSegs([])} className="px-3 py-1 text-[13px] text-[#EF4444] font-semibold">
            Clear
          </button>
        )}
      </div>

      {/* Probability histogram */}
      <SectionHeading>Churn Probability Distribution</SectionHeading>
      <ChartCard caption="Each bar shows how many customers fall in that probability range. A spike near 100% = high-confidence churn predictions.">
        <ResponsiveContainer width="100%" height={440}>
          <AreaChart data={histData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
            <defs>
              <linearGradient id="cg1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4F46E5" stopOpacity={0.85} />
                <stop offset="100%" stopColor="#7C3AED" stopOpacity={0.15} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis dataKey="range" tick={{ fontSize: 11, fill: "#6B7280" }} angle={-45} textAnchor="end" />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
            <Area type="monotone" dataKey="count" fill="url(#cg1)" stroke="#4F46E5" strokeWidth={2} name="Customers" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* Risk Tier by Segment */}
      <SectionHeading>Risk Tier by Segment</SectionHeading>
      <ChartCard caption="High Risk = churn probability above 60%. Use this to size retention budget by segment.">
        <ResponsiveContainer width="100%" height={440}>
          <BarChart data={riskBySegData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis dataKey="segment" tick={{ fontSize: 13, fill: "#6B7280" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 12 }} />
            <Bar dataKey="High Risk" fill="#EF4444" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Medium Risk" fill="#F59E0B" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Low Risk" fill="#10B981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* SHAP Feature Importance */}
      <SectionHeading>Top Churn Drivers (SHAP)</SectionHeading>
      <ChartCard caption="Mean absolute SHAP value across filtered customers — higher = stronger influence on churn prediction.">
        <ResponsiveContainer width="100%" height={440}>
          <BarChart
            data={shapData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 140, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis type="number" tick={{ fontSize: 12, fill: "#6B7280" }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 12, fill: "#1E1B4B" }} width={130} />
            <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
            <Bar dataKey="importance" name="Mean |SHAP|" radius={[0, 4, 4, 0]}>
              {shapData.map((_, i) => (
                <Cell key={i} fill={i === 0 ? "#4F46E5" : i === 1 ? "#6366F1" : "#818CF8"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
