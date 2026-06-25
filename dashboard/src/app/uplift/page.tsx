import {
  getUpliftKpis,
  getCustomerTypeSummary,
  getRoiBySegment,
  getTopPersuadables,
  getUpliftScatterData,
} from "@/lib/data";
import { UpliftClient } from "@/components/pages/uplift-client";

export const dynamic = "force-dynamic";

const EMPTY_KPIS = { persuadable_count: 0, positive_roi_count: 0, avg_uplift_score: 0, total_roi_potential: 0 };

export default async function UpliftPage() {
  const [kpis, typeSummary, roiBySeg, topPersuadables, scatter] = await Promise.all([
    getUpliftKpis().catch(() => EMPTY_KPIS),
    getCustomerTypeSummary().catch(() => []),
    getRoiBySegment().catch(() => []),
    getTopPersuadables(2000).catch(() => []),
    getUpliftScatterData().catch(() => []),
  ]);
  return (
    <UpliftClient
      kpis={kpis}
      typeSummary={typeSummary}
      roiBySeg={roiBySeg}
      topPersuadables={topPersuadables}
      scatter={scatter}
    />
  );
}
