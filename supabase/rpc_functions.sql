-- ============================================================
-- Subscription Churn Engine — Supabase RPC Functions
-- Run this once in: Supabase Dashboard → SQL Editor → Run
-- These server-side functions aggregate data before sending it
-- to the browser, bypassing the 1000-row max_rows cap entirely.
-- ============================================================

-- 1. Segment summary — everything the Segmentation page needs except scatter
CREATE OR REPLACE FUNCTION get_segment_summary()
RETURNS TABLE (
  segment             TEXT,
  customer_count      BIGINT,
  churn_rate          FLOAT,
  avg_churn_prob      FLOAT,
  high_risk_pct       FLOAT,
  persuadable_pct     FLOAT,
  avg_tenure          FLOAT,
  avg_satisfaction    FLOAT,
  avg_days_since_order FLOAT,
  avg_hour_spend      FLOAT,
  avg_cashback        FLOAT,
  gmm_high            BIGINT,
  gmm_medium          BIGINT,
  gmm_boundary        BIGINT
) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    segment,
    COUNT(*) AS customer_count,
    AVG(churn::FLOAT) AS churn_rate,
    AVG(churn_probability) AS avg_churn_prob,
    AVG(CASE WHEN risk_tier = 'High Risk'   THEN 1.0 ELSE 0.0 END) AS high_risk_pct,
    AVG(CASE WHEN customer_type = 'Persuadable' THEN 1.0 ELSE 0.0 END) AS persuadable_pct,
    AVG(tenure) AS avg_tenure,
    AVG(satisfaction_score) AS avg_satisfaction,
    AVG(days_since_last_order) AS avg_days_since_order,
    AVG(hour_spend_on_app) AS avg_hour_spend,
    AVG(cashback_amount) AS avg_cashback,
    SUM(CASE WHEN GREATEST(COALESCE(gmm_prob_seg0,0),COALESCE(gmm_prob_seg1,0),COALESCE(gmm_prob_seg2,0),COALESCE(gmm_prob_seg3,0),COALESCE(gmm_prob_seg4,0)) >= 0.9  THEN 1 ELSE 0 END) AS gmm_high,
    SUM(CASE WHEN GREATEST(COALESCE(gmm_prob_seg0,0),COALESCE(gmm_prob_seg1,0),COALESCE(gmm_prob_seg2,0),COALESCE(gmm_prob_seg3,0),COALESCE(gmm_prob_seg4,0)) >= 0.8
              AND GREATEST(COALESCE(gmm_prob_seg0,0),COALESCE(gmm_prob_seg1,0),COALESCE(gmm_prob_seg2,0),COALESCE(gmm_prob_seg3,0),COALESCE(gmm_prob_seg4,0)) < 0.9   THEN 1 ELSE 0 END) AS gmm_medium,
    SUM(CASE WHEN GREATEST(COALESCE(gmm_prob_seg0,0),COALESCE(gmm_prob_seg1,0),COALESCE(gmm_prob_seg2,0),COALESCE(gmm_prob_seg3,0),COALESCE(gmm_prob_seg4,0)) < 0.8   THEN 1 ELSE 0 END) AS gmm_boundary
  FROM customers
  GROUP BY segment
  ORDER BY customer_count DESC;
$$;

-- 2. Churn KPIs (supports optional segment filter)
CREATE OR REPLACE FUNCTION get_churn_kpis(p_segment TEXT DEFAULT NULL)
RETURNS TABLE (
  total           BIGINT,
  high_risk       BIGINT,
  avg_churn_prob  FLOAT,
  actual_churners BIGINT
) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    COUNT(*)                                                              AS total,
    SUM(CASE WHEN risk_tier = 'High Risk' THEN 1 ELSE 0 END)            AS high_risk,
    AVG(churn_probability)                                                AS avg_churn_prob,
    SUM(churn)                                                            AS actual_churners
  FROM customers
  WHERE p_segment IS NULL OR segment = p_segment;
$$;

-- 3. Churn probability histogram (supports optional segment filter)
CREATE OR REPLACE FUNCTION get_churn_histogram(p_segment TEXT DEFAULT NULL)
RETURNS TABLE (bucket INT, count BIGINT) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    LEAST(FLOOR(churn_probability * 10)::INT, 9) AS bucket,
    COUNT(*) AS count
  FROM customers
  WHERE p_segment IS NULL OR segment = p_segment
  GROUP BY bucket
  ORDER BY bucket;
$$;

-- 4. Risk tier counts per segment
CREATE OR REPLACE FUNCTION get_risk_summary()
RETURNS TABLE (
  segment     TEXT,
  high_risk   BIGINT,
  medium_risk BIGINT,
  low_risk    BIGINT
) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    segment,
    SUM(CASE WHEN risk_tier = 'High Risk'   THEN 1 ELSE 0 END) AS high_risk,
    SUM(CASE WHEN risk_tier = 'Medium Risk' THEN 1 ELSE 0 END) AS medium_risk,
    SUM(CASE WHEN risk_tier = 'Low Risk'    THEN 1 ELSE 0 END) AS low_risk
  FROM customers
  GROUP BY segment
  ORDER BY segment;
$$;

-- 5. SHAP feature importance (supports optional segment filter)
CREATE OR REPLACE FUNCTION get_shap_summary(p_segment TEXT DEFAULT NULL)
RETURNS TABLE (feature TEXT, avg_importance FLOAT) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    kv.key                            AS feature,
    AVG(ABS(kv.value::text::float))   AS avg_importance
  FROM customers c,
    jsonb_each(c.top_shap_features) kv
  WHERE (p_segment IS NULL OR c.segment = p_segment)
    AND c.top_shap_features IS NOT NULL
    AND c.top_shap_features != 'null'::jsonb
  GROUP BY kv.key
  ORDER BY avg_importance DESC
  LIMIT 8;
$$;

-- 6. Average churn probability per segment
CREATE OR REPLACE FUNCTION get_avg_churn_by_segment()
RETURNS TABLE (segment TEXT, avg_churn_prob FLOAT) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT segment, AVG(churn_probability) AS avg_churn_prob
  FROM customers
  GROUP BY segment
  ORDER BY avg_churn_prob DESC;
$$;

-- 7. Customer type distribution (uplift page)
CREATE OR REPLACE FUNCTION get_customer_type_summary()
RETURNS TABLE (
  customer_type       TEXT,
  count               BIGINT,
  avg_uplift_score    FLOAT,
  avg_net_roi         FLOAT,
  positive_roi_count  BIGINT,
  avg_churn_prob      FLOAT
) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    customer_type,
    COUNT(*)                                                      AS count,
    AVG(uplift_score)                                             AS avg_uplift_score,
    AVG(net_roi)                                                  AS avg_net_roi,
    SUM(CASE WHEN roi_positive THEN 1 ELSE 0 END)                AS positive_roi_count,
    AVG(churn_probability)                                        AS avg_churn_prob
  FROM customers
  GROUP BY customer_type
  ORDER BY count DESC;
$$;

-- 8. Avg ROI by segment for Persuadables only (uplift page)
CREATE OR REPLACE FUNCTION get_roi_by_segment()
RETURNS TABLE (segment TEXT, avg_roi FLOAT, persuadable_count BIGINT) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT segment, AVG(net_roi) AS avg_roi, COUNT(*) AS persuadable_count
  FROM customers
  WHERE customer_type = 'Persuadable'
  GROUP BY segment
  ORDER BY avg_roi DESC;
$$;

-- 9. Top N persuadables by net ROI (uplift + retention pages)
CREATE OR REPLACE FUNCTION get_top_persuadables(p_limit INT DEFAULT 200)
RETURNS TABLE (
  customer_id           TEXT,
  segment               TEXT,
  churn_probability     FLOAT,
  uplift_score          FLOAT,
  net_roi               FLOAT,
  intervention_priority INT
) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT customer_id, segment, churn_probability, uplift_score, net_roi, intervention_priority
  FROM customers
  WHERE customer_type = 'Persuadable'
  ORDER BY net_roi DESC
  LIMIT p_limit;
$$;

-- 10. Overall uplift KPIs
CREATE OR REPLACE FUNCTION get_uplift_kpis()
RETURNS TABLE (
  persuadable_count   BIGINT,
  positive_roi_count  BIGINT,
  avg_uplift_score    FLOAT,
  total_roi_potential FLOAT
) LANGUAGE SQL SECURITY DEFINER AS $$
  SELECT
    SUM(CASE WHEN customer_type = 'Persuadable' THEN 1 ELSE 0 END)         AS persuadable_count,
    SUM(CASE WHEN roi_positive THEN 1 ELSE 0 END)                           AS positive_roi_count,
    AVG(uplift_score)                                                         AS avg_uplift_score,
    SUM(CASE WHEN customer_type = 'Persuadable' AND roi_positive THEN net_roi ELSE 0 END) AS total_roi_potential
  FROM customers;
$$;
