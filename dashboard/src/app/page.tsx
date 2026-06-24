import { getSegmentSummary, getCustomers } from "@/lib/data";
import { SegmentationClient } from "@/components/pages/segmentation-client";

export const dynamic = "force-dynamic";

export default async function SegmentationPage() {
  const [summary, customers] = await Promise.all([
    getSegmentSummary(),
    getCustomers(),
  ]);
  return <SegmentationClient summary={summary} customers={customers} />;
}
