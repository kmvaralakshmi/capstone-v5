# V5 Research Benchmarks

This folder is intentionally a benchmark-input area, not a source of fabricated scores.

## NSE Top 200
Use the latest official NSE ESG research dataset/report and create `nse_top200.csv` with at least:
`Company_Name,Company_Code,NSE_ESG_Score` (plus E/S/G columns if available).

## NSE/CFA/CFA Society India
Use the published BRSR research for disclosure-quality comparison. This is not assumed to be a universal 0-100 ESG score. Create `nse_cfa_brsr.csv` using fields such as:
`Company_Code,Metric_Name,Disclosure_Status,Reported_Value,Assurance_Status,Value_Chain_Assessment`.

The V5 validation layer compares only common, explicitly comparable fields and reports missing/unavailable data rather than inventing values.
