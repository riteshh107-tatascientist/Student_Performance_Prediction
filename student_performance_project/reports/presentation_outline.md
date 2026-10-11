# Presentation Outline (8 slides)

1. **Title and objective** — Student Performance Analytics & Early Prediction; author, course, internship context.
2. **Problem and intended user** — Why early supportive insight may help; define prediction target `G3`.
3. **Dataset and ethics** — UCI dataset, 2 Portuguese secondary schools, mathematics subset, provenance and limits.
4. **Data workflow** — Load, validate, clean, EDA, encode/scale, train/test split.
5. **Exploratory findings** — Insert grade distribution and study-time chart from the app; only state observed patterns.
6. **Modeling and evaluation** — Baseline vs Ridge / Random Forest / Histogram Gradient Boosting; include actual MAE, RMSE, R² and 5-fold CV MAE.
7. **Dashboard demo** — Explore data, compare metrics, try a hypothetical profile; explain uncertainty.
8. **Conclusion and next steps** — measured outcome, limitations, fairness checks, local validation, possible future work.

**Important:** Fill the results slide only after running `python train.py`. Do not use invented metrics.
