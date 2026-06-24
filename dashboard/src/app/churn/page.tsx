import { getCustomers } from "@/lib/data";
import { ChurnClient } from "@/components/pages/churn-client";

export const revalidate = 3600;

export default async function ChurnPage() {
  const customers = await getCustomers();
  return <ChurnClient customers={customers} />;
}
