# Impulse Buying Behaviour

## Files
- `train_models.py` – audits the datasets, builds the target, compares models,
  and writes evaluation reports
- `Untitled form.csv` – Google Forms responses
- `social_media_buying.csv` – social media buying data
- `Cleaned dataset.xlsx` – psychological drivers data
- `requirements.txt` – Python libraries needed
- `results` – metrics, audit reports, graphs and the saved model

## Run the project

1. Open this folder in VS Code.
2. Open the terminal.
3. Install the libraries:

```bash
pip install -r requirements.txt
```

4. Obtain the original datasets through an authorized, private channel and place
   these files beside `train_models.py`:

   - `Untitled form.csv`
   - `social_media_buying.csv`
   - `Cleaned dataset.xlsx`

   They are intentionally excluded from this public repository because they
   contain respondent-level data. Do not commit or publicly upload them.

5. Run the program:

```bash
python train_models.py
```

6. Check the locally generated `results` folder.

The program validates the three source datasets and creates `impulse_target`:

- Google Forms: `Yes` = 0; `No` and `I was considering buying something
  similar` = 1.
- Social Media Buying: uses the validated `impulse_purchase` values (0 or 1).
- Psychological Drivers: requires all four `IB1`-`IB4` answers; their mean
  defines 1 when it is at least 4, otherwise 0.

The Social Media CSV contains pre-existing `impulse_purchase` labels. No
questionnaire, codebook, or target-generation script is present in the available
project files, so the rule used to create those labels cannot be established
from this project. The high Social Media scores must therefore be interpreted
with that provenance limitation in mind.

It compares Logistic Regression, Random Forest, Gradient Boosting, Extra Trees
and a Soft Voting Classifier. A stratified 80:20 split and five-fold stratified
cross-validation are used; preprocessing is fitted inside each training
pipeline, and model selection uses cross-validation accuracy rather than test
accuracy. The script also compares results with and without the source indicator,
audits Social Media predictor associations, and checks repeated stratified
splits.

The available files do not document a sufficiently comparable behavioral
feature set across all three questionnaires. Therefore, the script records why
the common-feature and leave-one-source-out experiments were not performed,
instead of reporting misleading results. The extended experiment keeps valid
source-specific features; its mixed-source metrics should not be interpreted as
proof of generalization to a new source.

Main local outputs include `unified_model_comparison.csv`,
`unified_source_test_performance.csv`, `unified_best_model.joblib`,
`unified_confusion_matrix.png`, `unified_accuracy_comparison.png`,
`dataset_summary.csv`, `target_distribution.csv`,
`feature_exclusion_report.csv`, `leakage_check_report.csv`,
`source_indicator_ablation.csv`, `social_media_leakage_sensitivity.csv`,
`social_media_repeated_split_results.csv`, and `final_model_report.txt`.

The audited run selected a Soft Voting Classifier made of Logistic Regression,
Random Forest, and Gradient Boosting. Its held-out accuracy was 79.58% (191 of
240 records correct). The Social Media subset was 87/87 correct, but standalone
repeated stratified splits ranged from 94% to 99%; neither result establishes
how the pre-existing target labels were generated or proves performance on
future data. Removing the four most target-associated Social Media features
together reduced the seed-42 sensitivity split accuracy to 59%, but this is an
exploratory ablation, not an independent estimate: those features were flagged
after examining their target associations. The Psychological Drivers subset
had an F1-score of 51.35%.
See `final_model_report.txt` for the confusion matrices, feature associations,
split details, and limitations.

The `results` folder may also contain older per-dataset outputs from the
previous training workflow, including `ALL_MODEL_RESULTS.csv`, the
`*_model_comparison.csv` files, per-dataset plots, and per-dataset model
artifacts. The current unified script does not generate or refresh those
legacy files; do not treat them as results from the audited unified run.

For inference, provide all feature columns expected by the saved pipeline
(48 columns, including `source_dataset`), with source-specific fields left
missing when unavailable. The pipeline imputes those missing values.

Original datasets, generated results, saved models, virtual environments, and
Python caches are excluded by `.gitignore`; they are not part of this public
repository.
