# SHAP Module

This folder contains the final delivery for Member D's SHAP interpretability task.

## Structure

- `code/D.py`: SHAP analysis script aligned with Member C's final GRU configuration.
- `results/shap_summary.png`: Global SHAP summary plot.
- `results/shap_dependence.png`: SHAP dependence plot for the most important feature.
- `results/shap_feature_importance.csv`: Feature importance ranking based on mean absolute SHAP values.
- `report/shap_report_zh.docx`: Chinese report.
- `report/shap_report_en.docx`: English report.

## Inputs

The SHAP analysis uses the following existing project files:

- `../processed/final_dataset_v2_plus.csv`
- `../GRU/results/member_C_results_selected.txt`
- `../GRU/results/best_model_C_selected.pth`

## How To Run

Run the SHAP analysis from the project root:

```powershell
.\.venv\Scripts\python.exe SHAP\code\D.py
```

## Final Deliverables

The final submission files for this module are:

- `SHAP/results/shap_summary.png`
- `SHAP/results/shap_dependence.png`
- `SHAP/results/shap_feature_importance.csv`
- `SHAP/report/shap_report_zh.docx`
- `SHAP/report/shap_report_en.docx`

## Notes

- The SHAP results are generated using the final GRU model selected by Member C.
- The current folder structure separates code, result artifacts, and written reports for easier review.
