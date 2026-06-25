import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-key";

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type Customer = {
  customer_id: string;
  segment: string;
  churn_probability: number;
  risk_tier: string;
  uplift_score: number;
  customer_type: string;
  net_roi: number;
  roi_positive: boolean;
  intervention_priority: number | null;
  umap_1: number;
  umap_2: number;
  tenure: number | null;
  satisfaction_score: number | null;
  days_since_last_order: number | null;
  hour_spend_on_app: number | null;
  complain: number | null;
  order_count: number | null;
  cashback_amount: number | null;
  churn: number;
  top_shap_features: Record<string, number> | null;
  gmm_prob_seg0: number | null;
  gmm_prob_seg1: number | null;
  gmm_prob_seg2: number | null;
  gmm_prob_seg3: number | null;
  gmm_prob_seg4: number | null;
};

export type RetentionAction = {
  id: string;
  customer_id: string;
  segment: string | null;
  churn_probability: number | null;
  uplift_score: number | null;
  net_roi: number | null;
  intervention_type: string | null;
  channel: string | null;
  timing: string | null;
  message_framing: string | null;
  agent_reasoning: unknown;
  agentic_mode: boolean;
  generated_at: string;
  outcome?: string | null;
};
