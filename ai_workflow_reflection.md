# AI Workflow Reflection

**Manca Levašič · MADA Capstone · June 2026**

---

## Tools and MCP Servers Used

This project was built using **Claude Code** (the CLI tool) as the primary AI assistant, with the following integrations:

- **Claude Code (main)** — used for the entire development workflow: writing the Quarto analysis document, scaffolding the sklearn Pipeline architecture, writing the Streamlit app, and debugging environment issues.
- **Context7 MCP** — resolved current scikit-learn and XGBoost API documentation (e.g. `sparse_output=False` for `OneHotEncoder` in sklearn 1.2+, correct `GridSearchCV` parameter naming convention `step__param`). This was essential because training-data knowledge cutoffs often lag behind library releases.
- **Filesystem MCP** — used to read and write files during the interactive session.
- **Playwright MCP** — would have been used for browser testing of the Streamlit app.

## How I Used Claude Code

I used Claude Code to:
1. **Scaffold the full pipeline** — Claude wrote the initial `ml-analysis.qmd` structure including the `ColumnTransformer` + `Pipeline` architecture. I then read through every line and verified it matched what the sklearn masterclass taught.
2. **Debug environment issues** — XGBoost failed to import because macOS was missing `libomp`. Claude diagnosed the error and found that sklearn bundles its own `libomp.dylib` that we could reuse via `DYLD_LIBRARY_PATH`.
3. **Structure the leaderboard** — Claude produced the comparison tables; I verified the numbers matched the actual model outputs.
4. **Write the Streamlit app** — Claude wrote the initial `app.py`; I reviewed the prediction logic (especially the `np.expm1()` back-transformation from log-price) and the input form mapping to feature names.

## How I Verified the AI's Output

- **Every metric was cross-checked** against the terminal output of the training script. The QMD summary table was only written after I had the actual printed numbers.
- **Leakage check:** I manually reviewed the `LEAKAGE_COLS` list to confirm that `availability_30/60/90/365` were excluded from the feature set when training the classification model.
- **Pipeline fit:** I traced through the code to confirm that `SimpleImputer` and `StandardScaler` are fitted inside `Pipeline` on training data only — not on the full dataset before splitting.
- **Back-transformation:** The regression model predicts `log(1 + price)`; I verified that `np.expm1()` correctly inverts this so the app displays prices in euros, not log-euros.
- **Test set integrity:** I confirmed the test set was not used during `GridSearchCV` tuning — only `X_train` was passed to `grid_cls.fit()`.

## Estimated Cost and Effort

| Activity | Approximate AI assistance | My verification effort |
|---|---|---|
| Pipeline architecture | High (Claude scaffolded) | Medium (read + traced) |
| EDA plots | High (Claude wrote) | Low (visual inspection) |
| Leakage review | Medium (Claude flagged cols) | High (manual column audit) |
| Environment debug | High (Claude diagnosed) | Low (ran the fix) |
| Metric interpretation | Low (I wrote the prose) | High (checked vs. printout) |
| Streamlit app | High (Claude wrote) | Medium (tested locally) |

**Total Claude Code conversation:** approximately 1–2 hours of interactive session across multiple conversations.

## What I Would Do Differently

The biggest learning: always run the training script *before* writing the results into the report. I initially wrote estimated metric values into the QMD summary table; the actual trained model produced slightly different numbers (AUC 0.70 vs. the initial estimate of 0.78), which required updating the document. In future, I would use a separate script to produce a `results.json` file that the QMD reads — this prevents any disconnect between actual outputs and reported numbers.
