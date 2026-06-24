"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Customer } from "@/lib/supabase";
import { PageTitle, SectionHeading } from "@/components/ui/section-heading";
import { MetricCard } from "@/components/ui/metric-card";
import { ChartCard } from "@/components/ui/chart-card";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from "recharts";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const SEGMENT_COLORS: Record<string, string> = {
  "Champions": "#4F46E5",
  "Loyal Customers": "#7C3AED",
  "At-Risk": "#EF4444",
  "Hibernating": "#F59E0B",
  "Lost Customers": "#6B7280",
};

const COLOR_OPTIONS = ["Segment", "Churn", "RiskTier", "ChurnProbability", "UpliftScore"];

function mapColor(c: Customer, by: string): string {
  if (by === "Segment") return SEGMENT_COLORS[c.segment] ?? "#6B7280";
  if (by === "Churn") return c.churn === 1 ? "#EF4444" : "#10B981";
  if (by === "RiskTier") {
    if (c.risk_tier === "High Risk") return "#EF4444";
    if (c.risk_tier === "Medium Risk") return "#F59E0B";
    return "#10B981";
  }
  return "#4F46E5";
}

interface Props { customers: Customer[] }

export function SegmentationClient({ customers }: Props) {
  const [colorBy, setColorBy] = useState("Segment");

  const segments = useMemo(() => {
    const map: Record<string, Customer[]> = {};
    for (const c of customers) {
      if (!map[c.segment]) map[c.segment] = [];
      map[c.segment].push(c);
    }
    return map;
  }, [customers]);

  const kpiData = useMemo(() =>
    Object.entries(segments).map(([seg, rows]) => ({
      segment: seg,
      count: rows.length,
      churnRate: rows.filter((r) => r.churn === 1).length / rows.length,
      color: SEGMENT_COLORS[seg] ?? "#6B7280",
    })), [segments]);

  // UMAP scatter
  const umapTraces = useMemo(() => {
    if (colorBy === "Segment") {
      return Object.entries(segments).map(([seg, rows]) => ({
        type: "scatter" as const,
        mode: "markers" as const,
        name: seg,
        x: rows.map((r) => r.umap_1),
        y: rows.map((r) => r.umap_2),
        text: rows.map((r) => `${r.customer_id}<br>Churn: ${(r.churn_probability * 100).toFixed(1)}%`),
        marker: { size: 5, color: SEGMENT_COLORS[seg] ?? "#6B7280", line: { width: 0.3, color: "white" } },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      }));
    }
    const colorScale = colorBy === "Churn"
      ? customers.map((c) => c.churn)
      : colorBy === "ChurnProbability"
        ? customers.map((c) => c.churn_probability)
        : customers.map((c) => c.uplift_score);
    return [{
      type: "scatter" as const,
      mode: "markers" as const,
      name: colorBy,
      x: customers.map((c) => c.umap_1),
      y: customers.map((c) => c.umap_2),
      marker: {
        size: 5,
        color: colorScale,
        colorscale: "RdYlGn_r",
        showscale: true,
        line: { width: 0.3, color: "white" },
      },
      text: customers.map((c) => `${c.customer_id}<br>Seg: ${c.segment}`),
      hovertemplate: "%{text}<extra></extra>",
    }];
  }, [colorBy, customers, segments]);

  // GMM confidence breakdown
  const gmmData = useMemo(() => {
    const segs = Object.keys(segments);
    return segs.map((seg) => {
      const rows = segments[seg];
      const confs = rows.map((r) => {
        const probs = [r.gmm_prob_seg0, r.gmm_prob_seg1, r.gmm_prob_seg2, r.gmm_prob_seg3, r.gmm_prob_seg4]
          .filter((v): v is number => v !== null);
        return probs.length ? Math.max(...probs) : 1;
      });
      return {
        segment: seg,
        "High (≥90%)": confs.filter((v) => v >= 0.9).length,
        "Medium (80-90%)": confs.filter((v) => v >= 0.8 && v < 0.9).length,
        "Boundary (<80%)": confs.filter((v) => v < 0.8).length,
      };
    });
  }, [segments]);

  // Heatmap data
  const heatmapData = useMemo(() => {
    const features = ["tenure", "satisfaction_score", "days_since_last_order", "hour_spend_on_app", "cashback_amount"];
    const labels = ["Tenure", "Satisfaction", "Days Since Order", "App Hours", "Cashback"];
    const segs = Object.keys(segments);
    return { features, labels, segs, values: features.map((f) =>
      segs.map((s) => {
        const rows = segments[s];
        const vals = rows.map((r) => (r as Record<string, unknown>)[f] as number).filter((v) => v != null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      })
    )};
  }, [segments]);

  return (
    <div>
      <PageTitle>Customer Segmentation</PageTitle>

      {/* KPI row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
        {kpiData.map((k) => (
          <MetricCard
            key={k.segment}
            label={k.segment}
            value={k.count.toLocaleString()}
            delta={`${(k.churnRate * 100).toFixed(1)}% churn`}
            accentColor={k.color}
          />
        ))}
      </div>

      {/* UMAP */}
      <SectionHeading>Customer Behavioral Space (2D UMAP Projection)</SectionHeading>
      <div className="flex items-center gap-3 mb-4">
        <label className="text-[13px] font-semibold text-[#4F46E5]">Colour by:</label>
        <select
          value={colorBy}
          onChange={(e) => setColorBy(e.target.value)}
          className="rounded-xl border-2 border-[#818CF8] bg-white px-3 py-2 text-[14px] text-[#1E1B4B] font-medium min-w-[180px] focus:outline-none focus:border-[#4F46E5]"
        >
          {COLOR_OPTIONS.map((o) => <option key={o}>{o}</option>)}
        </select>
      </div>
      <ChartCard caption="Each dot is a customer. UMAP reveals natural behavioral clusters that K-Means then labels.">
        <Plot
          data={umapTraces as Plotly.Data[]}
          layout={{
            height: 600,
            template: "plotly_white" as Plotly.Template,
            margin: { l: 30, r: 30, t: 30, b: 30 },
            legend: { orientation: "h", y: 1.05, x: 0 },
            paper_bgcolor: "white",
            plot_bgcolor: "white",
            font: { family: "Inter, sans-serif", color: "#334155" },
          }}
          config={{ responsive: true, displayModeBar: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </ChartCard>

      <div className="h-8" />

      {/* Heatmap */}
      <SectionHeading>Segment Feature Heatmap</SectionHeading>
      <ChartCard caption="Average feature value per segment — shows what makes each cluster behaviourally distinct.">
        <Plot
          data={heatmapData.values.map((row, fi) => ({
            type: "bar" as const,
            name: heatmapData.labels[fi],
            x: heatmapData.segs,
            y: row,
          }))}
          layout={{
            height: 460,
            barmode: "group" as const,
            template: "plotly_white" as Plotly.Template,
            margin: { l: 40, r: 20, t: 20, b: 80 },
            legend: { orientation: "h", y: -0.25 },
            paper_bgcolor: "white",
            plot_bgcolor: "white",
            font: { family: "Inter, sans-serif", color: "#334155" },
          }}
          config={{ responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </ChartCard>

      <div className="h-8" />

      {/* GMM Confidence */}
      <SectionHeading>Segment Assignment Confidence</SectionHeading>
      <ChartCard caption="Indigo = clearly placed (≥90%), amber = borderline, red = sits between two segments and warrants review.">
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={gmmData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E7FF" />
            <XAxis dataKey="segment" tick={{ fontSize: 13, fill: "#6B7280" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
            <Tooltip
              contentStyle={{ borderRadius: "10px", border: "2px solid #DDD6FE", fontSize: 13 }}
            />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 12 }} />
            <Bar dataKey="High (≥90%)" stackId="a" fill="#4F46E5" radius={[0, 0, 0, 0]} />
            <Bar dataKey="Medium (80-90%)" stackId="a" fill="#F59E0B" />
            <Bar dataKey="Boundary (<80%)" stackId="a" fill="#EF4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />

      {/* KPI table */}
      <SectionHeading>Segment Summary Table</SectionHeading>
      <div className="bg-white rounded-2xl border-2 border-[#DDD6FE] overflow-hidden shadow-sm">
        <table className="w-full text-[14px]">
          <thead>
            <tr style={{ background: "linear-gradient(110deg, #4338CA 0%, #7C3AED 100%)" }}>
              {["Segment", "Customers", "Actual Churn Rate", "Avg Churn Prob", "High Risk %", "Persuadable %"].map((h) => (
                <th key={h} className="text-white font-bold text-left px-4 py-3 text-[12px] uppercase tracking-wide">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {kpiData.map((k, i) => {
              const rows = segments[k.segment];
              const avgProb = rows.reduce((s, r) => s + r.churn_probability, 0) / rows.length;
              const highRisk = rows.filter((r) => r.risk_tier === "High Risk").length / rows.length;
              const persuadable = rows.filter((r) => r.customer_type === "Persuadable").length / rows.length;
              return (
                <tr key={k.segment} className={i % 2 === 0 ? "bg-white" : "bg-[#F5F3FF]"}>
                  <td className="px-4 py-3 font-semibold flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: k.color }} />
                    {k.segment}
                  </td>
                  <td className="px-4 py-3">{k.count.toLocaleString()}</td>
                  <td className="px-4 py-3">{(k.churnRate * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3">{(avgProb * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3">{(highRisk * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3">{(persuadable * 100).toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
