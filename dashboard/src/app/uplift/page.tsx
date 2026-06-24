import {
  getUpliftKpis,
  getCustomerTypeSummary,
  getRoiBySegment,
  getTopPersuadables,
  getCustomers,
} from "@/lib/data";
import { UpliftClient } from "@/components/pages/uplift-client";

export const dynamic = "force-dynamic";

export default async function UpliftPage() {
  const [kpis, typeSummary, roiBySeg, topPersuadables, customers] = await Promise.all([
    getUpliftKpis(),
    getCustomerTypeSummary(),
    getRoiBySegment(),
    getTopPersuadables(15),
    getCustomers(),
  ]);
  return (
    <UpliftClient
      kpis={kpis}
      typeSummary={typeSummary}
      roiBySeg={roiBySeg}
      topPersuadables={topPersuadables}
      customers={customers}
    />
  );
}
