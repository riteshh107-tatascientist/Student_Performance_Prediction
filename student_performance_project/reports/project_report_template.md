# Student Performance Analytics & Early Prediction — Project Report

**Author:** Ritesh Kumar Singh  
**Program:** CSE — Data Science  
**Project type:** Internship capstone / educational prototype  
**Dataset:** UCI Student Performance (Mathematics subset)

## 1. Executive summary
This project investigates whether student context, study habits, support indicators, and attendance can estimate final Mathematics grade (`G3`) before the final assessment. The workflow covers data acquisition, validation, exploratory analysis, preprocessing, model comparison, and a Streamlit application. Fill this section after running the code, using measured metrics only.

## 2. Problem statement
Educators may want early indicators to help prioritize supportive conversations and resources. The model estimates a grade; it does not diagnose motivation, ability, or personal circumstances.

## 3. Dataset and provenance
Cite: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. The UCI dataset contains records from two Portuguese secondary schools. The collection includes mathematics and Portuguese subjects; this project is scoped to mathematics. State the actual row/column counts from `artifacts/run_metadata.json` after running.

## 4. Data preparation and quality checks
Document: dataset dimensions, data types, missing-value counts, duplicate checks, target distribution, outliers, and any transformations. The pipeline uses median imputation and scaling for numeric fields and most-frequent imputation plus one-hot encoding for categorical fields. These transformations are learned inside the training pipeline.

## 5. Exploratory data analysis
Include charts for:
- Final grade distribution (`G3`).
- Average final grade by study-time category.
- Absences versus final grade.
- Correlation heatmap for numeric variables.

Describe patterns carefully as associations, not causes. Mention that the dataset's sample and collection context limit generalization.

## 6. Modeling methodology
Target: `G3` (0–20). Exclude `G1` and `G2` in the primary early-prediction model because the UCI documentation notes their strong correlation with `G3`. Use an 80/20 train/test split with `random_state=42`. Select among a mean baseline, Ridge regression, Random Forest, and Histogram Gradient Boosting using five-fold cross-validation MAE on the training set.

## 7. Results
Paste the actual `artifacts/model_metrics.csv` table here. Report test MAE, test RMSE, and test R² for each model. Explain that MAE is the average absolute grade error in points. Do not invent or round metrics before the run. Identify the selected model based on cross-validation MAE, not on repeatedly tuning against the test set.

## 8. Findings and recommendations
Write only findings supported by the EDA and evaluation. Potential responsible use is to offer voluntary, supportive resources and invite educators to review context—not to make automated decisions about individual students.

## 9. Limitations and ethics
- Only two Portuguese secondary schools are represented.
- Dataset may be old and may not represent current teaching contexts.
- Small sample size means performance estimates can vary.
- Features can encode sensitive social context and proxy variables.
- Predictions have uncertainty and do not establish causal relationships.
- This prototype is not approved for high-stakes decisions.

## 10. Reproducibility
Record Python and package versions, run command, random seed, dataset citation, and the actual generated metrics. Include Git commit hash if submitted from GitHub.

## 11. Conclusion
Summarize the measured performance and whether it improves over the baseline. State what additional validation is required before practical use.
