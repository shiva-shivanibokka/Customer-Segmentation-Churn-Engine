-- ============================================================
-- Subscription Churn Engine — Supabase Row-Level Security (RLS)
-- Run this once in: Supabase Dashboard → SQL Editor → Run
--
-- WHY THIS EXISTS
-- ---------------
-- The dashboard talks to Supabase with the public anon key, which
-- ships inside the browser bundle (NEXT_PUBLIC_SUPABASE_ANON_KEY).
-- That key is safe to expose ONLY when RLS is enabled — RLS is the
-- gate that defines what the anon role may actually do.
--
-- Without RLS, the public key grants full SELECT/INSERT/UPDATE/DELETE
-- on every exposed table, so anyone with the project URL could read,
-- overwrite, or delete all data. This script closes that hole while
-- keeping the dashboard fully functional.
--
-- ACCESS MODEL
-- ------------
--   • anon (browser):   READ-only on all data tables,
--                       plus INSERT on intervention_feedback
--                       (the one write the browser performs directly —
--                        see saveFeedback() in dashboard/src/lib/data.ts).
--   • service_role:     full access; used server-side only by the AI
--                       agent route (dashboard/src/app/api/agent/route.ts)
--                       and the Python pipeline. Bypasses RLS by design,
--                       so retention_actions writes are unaffected.
--   • RPC functions:    defined SECURITY DEFINER in rpc_functions.sql,
--                       so the aggregation RPCs run regardless of RLS.
--
-- Run AFTER the tables exist (config_tables.sql + the customers /
-- retention_actions / intervention_feedback tables) and AFTER
-- rpc_functions.sql. Re-running is safe — policies are dropped first.
-- ============================================================

-- ── 1. Enable RLS on every exposed table ────────────────────────────────────
ALTER TABLE customers              ENABLE ROW LEVEL SECURITY;
ALTER TABLE retention_actions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE intervention_feedback  ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_config        ENABLE ROW LEVEL SECURITY;
ALTER TABLE retention_playbook     ENABLE ROW LEVEL SECURITY;

-- ── 2. Public (anon) read access for what the dashboard renders ──────────────
DROP POLICY IF EXISTS anon_read_customers     ON customers;
DROP POLICY IF EXISTS anon_read_actions       ON retention_actions;
DROP POLICY IF EXISTS anon_read_feedback      ON intervention_feedback;
DROP POLICY IF EXISTS anon_read_config        ON business_config;
DROP POLICY IF EXISTS anon_read_playbook      ON retention_playbook;

CREATE POLICY anon_read_customers ON customers
  FOR SELECT TO anon USING (true);

CREATE POLICY anon_read_actions ON retention_actions
  FOR SELECT TO anon USING (true);

CREATE POLICY anon_read_feedback ON intervention_feedback
  FOR SELECT TO anon USING (true);

CREATE POLICY anon_read_config ON business_config
  FOR SELECT TO anon USING (true);

CREATE POLICY anon_read_playbook ON retention_playbook
  FOR SELECT TO anon USING (true);

-- ── 3. The only direct browser write: intervention feedback inserts ──────────
DROP POLICY IF EXISTS anon_insert_feedback ON intervention_feedback;

CREATE POLICY anon_insert_feedback ON intervention_feedback
  FOR INSERT TO anon WITH CHECK (true);

-- ── 4. (Intentional) No anon UPDATE/DELETE policies anywhere ─────────────────
-- With RLS enabled and no matching policy, UPDATE and DELETE from the anon
-- key are denied on all tables. retention_actions writes are performed
-- server-side with the service_role key, which bypasses RLS entirely.

-- ── 5. Verify ────────────────────────────────────────────────────────────────
-- Confirm RLS is on for all five tables:
--   SELECT relname, relrowsecurity
--   FROM pg_class
--   WHERE relname IN ('customers','retention_actions','intervention_feedback',
--                     'business_config','retention_playbook');
-- Confirm the policies exist:
--   SELECT tablename, policyname, cmd, roles FROM pg_policies ORDER BY tablename;
