# INN-Hotels-Cancellation-Prediction
Predict booking cancellations for a hotel chain. EDA, engineered features, logistic regression vs pruned decision tree, and ROC-based thresholding aligned to policy costs. Outputs: top predictors and actionable refund/overbooking insights.

**Commit message:** `Initial commit: ReCell pricing regression notebook and README`

---

## 4) INN Hotels

```markdown
# INN Hotels: Booking Cancellation Prediction

Predict booking cancellations early to inform refund policies and reduce lost revenue.

## Overview
- **Goal:** Early identification of likely cancellations.
- **Techniques:** EDA, preprocessing, logistic regression, decision trees with pruning, multicollinearity checks, ROC-AUC thresholding.

## Data
Hotel bookings dataset (not included).

## How to Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
