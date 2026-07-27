-- Run this once if 20260723_bei_streamlit_imports.sql was already executed
-- before the explicit Data API grants were added.

grant usage on schema public to service_role;
grant select, insert, update, delete on public.bei_import_batches to service_role;
grant select, insert, update, delete on public.bei_import_rows to service_role;
grant select, insert, update, delete on public.ai_analyses to service_role;
grant usage, select on sequence public.bei_import_rows_id_seq to service_role;
