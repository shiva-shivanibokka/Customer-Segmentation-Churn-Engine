"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  "Persuadable": "#4F46E5",
  "Sure Thing": "#10B981",
  "Lost Cause": "#EF4444",
  "Sleeping Dog": "#F59E0B",
};

interface Props { customers: Customer[] }

export function UpliftClient({ customers }: Props) {
  const kpis = useMemo(() => {
    const persuadable = customers.filter((c) => c.customer_type === "Persuadable").length;
    const positiveROI = customers.filter((c) => c.roi_positive).length;
    const avgUplift = customers.reduce((s, c) => s + c.uplift_score, 0) / customers.length;
    const totalROI = customers.filter((c) => c.roi_positive).reduce((s, c) => s + c.net_roi, 0);
    return { persuadable, positiveROI, avgUplift, totalROI };
  }, [customers]);

  // Uplift vs ChurnProb scatter (Persuadable quadrant)
  const byType = useMemo(() => {
    const types = [...new Set(customers.map((c) => c.customer_type))];
    return types.map((t) => {
      const rows = customers.filter((c) => c.customer_type === t);
      return {
        type: "scatter" as const,
        mode: "markers" as const,
        name: t,
        x: rows.map((c) => c.churn_probability),
        y: rows.map((c) => c.uplift_score),
        text: rows.map((c) => `${c.customer_id}<br>ROI: $${c.net_roi.toFixed(0)}`),
        marker: { size: 6, color: TYPE_COLORS[t] ?? "#6B7280", line: { width: 0.3, color: "white" } },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      };
    });
  }, [customers]);

  // ROI by segment
  const roiBySegment = useMemo(() => {
    const segs = [...new Set(customers.map((c) => c.segment))];
    return segs.map((seg) => {
      const rows = customers.filter((c) => c.segment === seg && c.customer_type === "Persuadable");
      return {
        segment: seg,
        avgROI: rows.length ? rows.reduce((s, c) => s + c.net_roi, 0) / rows.length : 0,
        count: rows.length,
      };
    }).sort((a, b) => b.avgROI - a.avgROI);
  }, [customers]);

  // Customer type distribution
  const typeDistribution = useMemo(() => {
    const types = [...new Set(customers.map((c) => c.customer_type))];
    return types.map((t) => ({
      type: t,
      count: customers.filter((c) => c.customer_type === t).length,
      color: TYPE_COLORS[t] ?? "#6B7280",
    }));
  }, [customers]);

  // Top Persuadables table
  const topPersuadables = useMemo(() =>
    customers
      .filter((c) => c.customer_type === "Persuadable")
      .sort((a, b) => b.net_roi - a.net_roi)
      .slice(0, 15),
    [customers]
  );

  return (
    <div>
      <PageTitle>Uplift Intelligence</PageTitle>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard label="Persuadables" value={kpis.persuadable.toLocaleString()} delta="Worth targeting" accentColor="#4F46E5" />
        <MetricCard label="Positive ROI" value={kpis.positiveROI.toLocaleString()} delta="Financially justified" accentColor="#10B981" />
        <MetricCard label="Avg Uplift" value={kpis.avgUplift >= 0 ? `+${(kpis.avgUplift * 100).toFixed(2)}%` : `${(kpis.avgUplift * 100).toFixed(2)}%`} accentColor="#7C3AED" />
        <MetricCard label="Total ROI Potential" value={`$${(kpis.totalROI / 1000).toFixed(0)}k`} accentColor="#F59E0B" />
      </div>

      {/* Uplift quadrant */}
      <SectionHeading>Uplift Quadrant — Who to Target</SectionHeading>
      <ChartCard caption="Persuadables (top-right): high churn risk AND respond to intervention — your primary target list.">
        <Plot
          data={byType as Plotly.Data[]}
          layout={{
            height: 520,
            template: "plotly_white" as Plotly.Template,
            xaxis: { title: "Churn Probability →", zeroline: true },
            yaxis: { title: "Uplift Score (T–S learner) →", zeroline: true },
            shapes: [
              { type: "line" as const, x0: 0.3, x1: 0.3, y0: -1, y1: 1, line: { dash: "dot", color: "#6B7280", width: 1.5 } },
              { type: "line" as const, x0: 0, x1: 1, y0: 0.05, y1: 0.05, line: { dash: "dot", color: "#6B7280", width: 1.5 } },
            ],
            margin: { l: 60, r: 30, t: 30, b: 60 },
            legend: { orientation: "h", y: 1.05, x: 0 },
            paper_bgcolor: "white",
            font: { family: "Inter, sans-serif", color: "#334155" },
          }}
          config={{ responsive: true, displayModeBar: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </ChartCard>

      <div className="h-8" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Customer type distribution */}
        <div>
          <SectionHeading>Customer Type Distribution</SectionHeading>
          <ChartCard>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={typeDistribution} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
                <XAxis dataKey="type" tick={{ fontSize: 12, fill: "#6B7280" }} />
                <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
                <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} />
                <Bar dataKey="count" name="Customers" radius={[4, 4, 0, 0]}>
                  {typeDistribution.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ROI by segment */}
        <div>
          <SectionHeading>Avg ROI by Segment (Persuadables)</SectionHeading>
          <ChartCard caption="Expected net return per intervention for Persuadables in each segment.">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={roiBySegment} layout="vertical" margin={{ top: 10, right: 30, left: 110, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
                <XAxis type="number" tickFormatter={(v) => `$${v.toFixed(0)}`} tick={{ fontSize: 11, fill: "#6B7280" }} />
                <YAxis type="category" dataKey="segment" tick={{ fontSize: 12, fill: "#1E1B4B" }} width={105} />
                <Tooltip contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }} formatter={(v) => [`$${Number(v).toFixed(2)}`, "Avg ROI"]} />
                <Bar dataKey="avgROI" fill="#4F46E5" radius={[0, 4, 4, 0]} name="Avg ROI" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>

      <div className="h-8" />

      {/* Top Persuadables table */}
      <SectionHeading>Top 15 Persuadables by ROI</SectionHeading>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)" }}>
              {["Customer ID", "Segment", "Churn Prob", "Uplift Score", "Net ROI", "Priority"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {topPersuadables.map((c, i) => (
              <tr key={c.customer_id} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                <td className="px-4 py-2.5 font-mono text-[13px] text-[#4F46E5] font-semibold">{c.customer_id}</td>
                <td className="px-4 py-2.5">{c.segment}</td>
                <td className="px-4 py-2.5">{(c.churn_probability * 100).toFixed(1)}%</td>
                <td className="px-4 py-2.5">{c.uplift_score >= 0 ? "+" : ""}{(c.uplift_score * 100).toFixed(2)}%</td>
                <td className="px-4 py-2.5 font-semibold text-[#10B981]">${c.net_roi.toFixed(2)}</td>
                <td className="px-4 py-2.5">{c.intervention_priority ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
