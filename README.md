# Banking AI campaign system

Portfolio project for **retail banking analytics and responsible AI**: predictive propensity, campaign targeting, and governance-style controls before any automated outreach narrative.

## What this repo contains

| Item | Description |
|------|-------------|
| `bank.csv` | Bank Marketing dataset (UCI / common Kaggle variant; term deposit subscription outcome). |
| `loan_model.ipynb` | End-to-end notebook: EDA, ML models, threshold and lift analysis, segment watchlist, guardrail actions, mock LLM offer drafts, and a one-call `run_end_to_end_demo()` helper. |

## Problem framing (banking context)

- **Level 1 — Propensity / response modelling:** Gradient Boosting and Random Forest–style workflows (trees + preprocessing) to rank customers by likelihood of a positive outcome, with holdout metrics.
- **Level 2 — Campaign design:** Threshold tuning and top-*N* (decile) lift to translate scores into **call-budget** and **conversion** trade-offs.
- **Level 3 — Governed personalization:** Segment risk table, `ALLOW` / `REVIEW` / `BLOCK` style gates, then **mock** LLM-generated copy (no API keys required) plus a simple text guardrail pass.

This mirrors how teams in markets such as **Bahrain** often separate **model quality**, **business policy**, and **compliance review** before customer-facing messaging.

## Setup

Python **3.10+** recommended (notebook tested with a 3.14 venv in development).

```bash
cd banking-ai-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook loan_model.ipynb
```

Run cells top to bottom, or jump to **Step 12** for `run_end_to_end_demo()` after dependencies are installed.

## Design choices worth calling out in interviews

- **Leakage control:** `duration` is excluded from training features when building a realistic “pre-contact” propensity story.
- **Stratified split** and **ROC-AUC / lift** reporting for imbalanced marketing outcomes.
- **Fairness watchlist** is a lightweight segment monitor (sample size + calibration-style gap), not a full legal fairness audit.
- **Mock LLM** is intentional: reproducible demos without storing secrets; swap `mock_llm_generate` for a real provider when you are ready.

## Dataset credit

Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository.  
[https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306)

If you obtained `bank.csv` via Kaggle, cite that source in derivative work as required by their terms.

## License

Notebook and code in this repository are provided as **portfolio / educational** material unless you add an explicit open-source license. Add a `LICENSE` file if you want others to reuse the code under clear terms.

---

**Author:** Jumana Iqbal
