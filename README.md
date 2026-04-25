# Project: Hotel Booking Cancellation Prediction

> **Context:** Capstone of the Supervised Learning – Classification module of my Applied Data Science program. INN Hotels is a simulated Portuguese hotel group losing significant revenue to last-minute booking cancellations — 32.8% of all 36,275 reservations (2017-2018) were being canceled, and those slots frequently couldn't be resold in time. I was asked to build a model that flags high-risk bookings early enough for the revenue team to intervene: a tighter deposit policy, targeted reminder emails, or selective overbooking. The project demonstrates decision-tree-based classification end-to-end — building a baseline tree, then a pruned / cost-complexity-tuned tree, then a logistic regression for interpretability comparison — along with confusion-matrix reasoning and the explicit trade-off between precision and recall when the business cost of a false negative (a missed cancellation) is much higher than a false positive. I use it for any conversation about classification, imbalanced-class thinking, or turning a model into an operational workflow.

---

## The problem

INN Hotels Group operates a chain of hotels across Portugal and faces a significant revenue challenge: approximately 32.8% of bookings are canceled, causing losses through room unavailability, increased distribution costs, and forced last-minute price reductions. The cancellation of bookings impacts the hotel on multiple fronts: loss of revenue when rooms cannot be resold, additional costs from distribution channel commissions, reduced profit margins from emergency pricing, and wasted human resources. Last-minute cancellations create the most severe impact. With a dataset of 36,275 bookings spanning 2017-2018, the goal was to build a predictive model that identifies which bookings are likely to be canceled in advance so the hotel can formulate profitable cancellation policies and take proactive retention measures.

---

## What I built

Built a classification model pipeline that predicts booking cancellations with an F1 score of 0.69 on the training set and balanced 0.68 F1 on test data. Trained two primary models—Logistic Regression and Decision Tree—on 36,275 hotel bookings with 18 features including lead time, special requests, meal plans, room pricing, guest demographics, and previous booking history. The preprocessed dataset was split 70/30 for train/test validation. The decision tree model (pre-pruned with max_depth constraint) emerged as the production choice, delivering balanced precision and recall with interpretable decision rules. Key predictor variables identified: lead_time (151 days threshold), number of special requests, average price per room, market segment (online vs. offline), and arrival month.

---

## Technical choices (and why)

- **Why classification models**: Binary classification (canceled/not canceled) with F1 score optimization to balance the cost of false negatives (lost resources) and false positives (damaged brand equity from overly conservative predictions).
- **Logistic Regression first**: Provides interpretable coefficients as log-odds. Found that lead_time, special_requests, required_car_parking_space, and meal_plan_selection had significant impact (p < 0.05). Revealed that each additional special request decreases cancellation odds by 77%.
- **Decision Tree for production**: Although initial trees overfit (100% training accuracy, 65% test F1), pre-pruning with max_depth constraints yielded balanced performance (0.69 F1 train, 0.68 F1 test) with human-readable rules for hotel managers.
- **Threshold optimization**: Used precision-recall and ROC-AUC curves to tune the classification threshold from default 0.5 to 0.42, achieving better balance between catching cancellations and not over-predicting (which damages service quality).
- **Multicollinearity handling**: Checked VIF on all numerical features; none showed moderate or high multicollinearity, so retained all significant variables.

---

## The outcome

- **Predictive performance**: F1 score of 0.69 on training set, 0.68 on test set with balanced precision (0.68) and recall (0.68) at 0.42 threshold.
- **Key drivers identified**: Lead time is the strongest predictor; bookings made >151 days in advance have much higher cancellation risk. Number of special requests inversely correlates with cancellation (customers who specify requests are 77% less likely to cancel).
- **Business actionability**: Decision tree rules are interpretable for operations teams. Example rule: "If lead time > 151 days AND avg price > 100 euros AND arrival month = December, then likely not canceled."
- **Retention lever**: Customers making special requests (43.5% of bookings) had significantly lower cancellation rates, suggesting post-booking engagement drives commitment.
- **Temporal pattern**: September and October see highest booking volume (~5000 each month) but also highest cancellation rates (~40%), while December/January see lowest cancellation rates (~20%), likely due to holiday travel commitment.

---

## What failed or what you tried

- **Initial decision tree overfitting**: Unpruned decision trees achieved 100% training accuracy but only 65% test accuracy and 0.58 F1 on test data. The model memorized training patterns without generalizing. Addressed by adding pre-pruning constraints (max_depth=10, min_samples_leaf=50).
- **Post-pruning complexity vs. interpretability trade-off**: Cost complexity pruning (alpha-based) produced higher F1 scores (0.74 on test) but with much larger, complex trees that operations teams couldn't use to make decisions. Pre-pruned simpler trees with lower metrics were more deployable.
- **Handling outliers**: Initially tried dropping extreme lead_time values (>500 days) and very high prices (>500 euros), but these represented real business cases (corporate bookings, far-advance holiday planning). Kept them in the final model.
- **Class imbalance**: Target variable was imbalanced (32.8% canceled, 67.2% not canceled). Tried balanced class weights in Logistic Regression and sklearn models to prevent bias toward the majority class, which improved recall but reduced precision slightly.

---

## Production and scale thinking

- **Deployment scenario**: The predictive model would be deployed as a real-time scoring service integrated with the hotel's reservation system, triggering automated interventions when a booking scores high for cancellation risk.
- **Volume**: With 36,275 annual bookings (across multiple Portugal locations), the model would score approximately 70-100 new bookings per day at peak seasons, requiring sub-100ms inference.
- **Monitoring**: Track prediction drift by monitoring the distribution of predicted cancellation scores over time and correlate against actual outcomes. If the cancellation rate suddenly shifts (e.g., external events like COVID, travel restrictions), retrain the model quarterly.
- **Maintenance**: Retrain model quarterly on rolling 2-year window to capture seasonal patterns. Update decision tree thresholds (currently 151 days, 100 euros) annually based on new booking behavior.
- **Intervention actions**: For high-risk bookings, trigger: (1) automated re-confirmation email 30 days before arrival, (2) offer to collect special requests (correlated with retention), (3) apply stricter cancellation policies with clear messaging at booking confirmation.

---

## Links

- GitHub: [link to repository if applicable]
- Notebook: [link to full analysis notebook]

---

## Executive Summary

> "I built a decision tree classifier that predicts hotel booking cancellations with 69% F1 score for a Portuguese hotel chain. By identifying lead time and special requests as the primary cancellation drivers, we helped the business implement targeted retention strategies—customers who make special requests are 77% less likely to cancel—and developed simple, interpretable decision rules that operations teams can use to flag high-risk bookings and intervene before revenue is lost."
