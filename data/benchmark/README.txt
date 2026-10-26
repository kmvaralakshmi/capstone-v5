BENCHMARK FILES (you supply the data; the code does the comparison)

1) nse_top200_esg.csv   <- copy nse_top200_esg_template.csv, fill it, save with this name.
   Needs a company identifier (Company_Code = NSE symbol, or Company_Name) and NSE_ESG_Score.
   NSE_E_Score / NSE_S_Score / NSE_G_Score are optional. Any scale works: scores whose maximum is
   <= 10 are multiplied by 10, so everything is compared on 0-100.

2) nse_cfa_reference.csv <- copy nse_cfa_reference_template.csv and fill Reference_Pct_Companies_Reporting
   (the % of companies the NSE/CFA study found reporting each metric).

Then run:   python run_agents.py --agents agent8
Outputs:    processed-data/validation_*.csv and validation_summary.json
