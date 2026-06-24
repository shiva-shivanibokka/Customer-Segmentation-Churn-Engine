import { getCustomers } from "@/lib/data";
import { RetentionClient } from "@/components/pages/retention-client";

export const dynamic = "force-dynamic";

export default async function RetentionPage() {
  const customers = await getCustomers();
  return <RetentionClient customers={customers} />;
}
