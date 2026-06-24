import { getCustomers } from "@/lib/data";
import { UpliftClient } from "@/components/pages/uplift-client";

export const revalidate = 3600;

export default async function UpliftPage() {
  const customers = await getCustomers();
  return <UpliftClient customers={customers} />;
}
