import { getRetentionActions, getAuditSummary } from "@/lib/data";
import { AnalyticsClient } from "@/components/pages/analytics-client";

export const revalidate = 0;

export default async function AnalyticsPage() {
  const [actions, summary] = await Promise.all([
    getRetentionActions(200),
    getAuditSummary(),
  ]);
  return <AnalyticsClient actions={actions} summary={summary} />;
}
