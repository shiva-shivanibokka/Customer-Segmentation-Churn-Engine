import { getCustomers } from "@/lib/data";
import { SegmentationClient } from "@/components/pages/segmentation-client";

export const revalidate = 3600;

export default async function SegmentationPage() {
  const customers = await getCustomers();
  return <SegmentationClient customers={customers} />;
}
