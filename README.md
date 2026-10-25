# Beauty Review Intelligence

Six week data science internship project by M Nithish Kumar.

**Status: executed synthetic demonstration and research-backed real-data plan.**
No actual Amazon customer reviews were downloaded or analyzed. The generated
dataset is intentionally artificial. Its performance is not evidence of
real-world effectiveness. This is a review-text classification project, not
a forecast of future purchases, a recommendation engine, or a medical tool.

## Business problem

Classify incoming beauty-product review text as likely low rating (1 or 2)
versus other rating (3 to 5) to support human feedback triage. Where a rating
already exists, use it directly; the text model is useful only for feedback
without ratings or for studying text-based triage. Cross-channel transfer
requires new human-labelled validation.

## What is included

- `reports/`: six separate Word reports and six submission descriptions.
- `src/run_project.py`: deterministic data generation, cleaning, EDA, two
  TF-IDF classifiers, validation threshold selection, and holdout evaluation.
- `data/`: raw and cleaned **synthetic** records, never real customers.
- `outputs/`: metrics, cleaning audit, partition manifest, predictions, charts,
  and exact package versions in the summary.
- `requirements-lock.txt`: versions used for the included run.
- `GITHUB_SETUP.md`: beginner-friendly publishing and submission instructions.

## Run on Windows

1. Extract the ZIP. Open the `beauty-review-intelligence` folder in VS Code:
   **File > Open Folder**.
2. Select **Terminal > New Terminal**. Check the terminal is in this folder.
3. Enter `py -m venv .venv`.
4. Enter `.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`.
5. Enter `.venv\Scripts\python.exe src\run_project.py`.
6. Open `outputs/model_metrics.csv` and the PNG charts.

Tested with Python 3.12.14. Python 3.12 is recommended for reproducing this
environment. Package installation needs internet access. If `py` is missing,
install Python and restart VS Code. No API keys or paid tools are required.

## Main design decisions

Rating defines the label and is never an input feature. Empty text and invalid
labels are removed rather than invented. Identical text with conflicting
binary labels is quarantined by exclusion; remaining identical texts are
deduplicated. This conservative **simulation policy** changes prevalence and
must be reconsidered on real data. A product-group split prevents product
overlap; assertions also check exact-text disjointness. TF-IDF learns only
from training data. Logistic regression and multinomial naive Bayes are
compared with a prior baseline. Validation F1 selects the model and threshold;
the holdout is evaluated only after that decision.

## Reproduce the included result

Seed 42 creates 1,820 raw rows including injected errors. Cleaning retains
1,316 rows. There are 803 training, 247 validation, and 266 test examples.
Logistic regression is selected on validation F1. At threshold 0.45 its test
precision is 0.8200, recall 0.7193, and F1 0.7664. These are **measured on
synthetic data only**. The bootstrap interval is a conditional row-resampling
demonstration and ignores product clustering and model-selection uncertainty.

## Real-data extension

Primary candidate: https://amazon-reviews-2023.github.io/main.html

Use the publisher's All Beauty review files after checking terms and access.
Record source version, retrieval date, SHA-256, sampling method and schema.
Map `rating`, `text`, `parent_asin` and `timestamp` to the project's schema;
convert timestamp units only after confirming the source schema. Do not run
the synthetic generator on a real input and call the resulting analysis real.
The current script has **no real-data import mode**. A reviewed adapter and
additional validation are future work.

The references and sourcing comparison in Week 2 cover alternative sources.
Public availability does not automatically permit redistribution. Do not
commit raw real reviews, reviewer identities, or credentials without review.

## Weekly deliverables

Week 1 defines the problem and hypotheses. Week 2 evaluates sources. Week 3
documents cleaning. Week 4 reports synthetic EDA and planned features. Week 5
compares models and validation. Week 6 evaluates measured synthetic results
and proposes a real-data pilot. The assignment permits simulation; whether
the internship accepts this scope is a decision for its evaluator.

## Limitations

Short generated phrases, constructed ratings, English-only text, no observed
customers, no causal conclusions and no measured financial benefit. Rating
is an imperfect proxy for sentiment. Real duplicates may be legitimate, and
three-star feedback may contain important complaints. Human review remains
necessary. No hours worked, deployment, employer results or real customer
outcomes are claimed.
