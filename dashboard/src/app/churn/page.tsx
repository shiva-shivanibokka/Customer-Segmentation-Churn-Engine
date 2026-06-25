import { getCustomers } from "@/lib/data";
import { ChurnClient } from "@/components/pages/churn-client";

export const dynamic = "force-dynamic";

export default async function ChurnPage() {
  const customers = await getCustomers().catch(() => []);
  return <ChurnClient customers={customers} />;
}
