# Impulse Buying Behaviour

A Python project for auditing three impulse-buying datasets, creating a
unified target, comparing machine-learning models, and generating evaluation
reports and visualisations.

## Repository file structure

```text
impulse_buying_student_version/
|-- train_models.py       # Dataset audit, target creation, training, evaluation
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation
|-- .gitignore            # Privacy and generated-file exclusions
|-- Untitled form.csv     # Local Google Forms data (not committed)
|-- social_media_buying.csv
|                          # Local social-media data (not committed)
|-- Cleaned dataset.xlsx  # Local psychological-drivers data (not committed)
|-- results/
|   |-- Untitled form.csv
|   |-- social_media_buying.csv
|   |-- Cleaned dataset.xlsx
|   |-- unified_model_comparison.csv
|   |-- unified_source_test_performance.csv
|   |-- unified_best_model.joblib
|   |-- unified_confusion_matrix.png
|   |-- unified_accuracy_comparison.png
|   |-- dataset_summary.csv
|   |-- target_distribution.csv
|   |-- feature_exclusion_report.csv
|   |-- leakage_check_report.csv
|   |-- source_indicator_ablation.csv
|   |-- social_media_leakage_sensitivity.csv
|   |-- social_media_repeated_split_results.csv
|   |-- final_model_report.txt
|   `-- *_model_comparison.csv, *_confusion_matrix.png,
|       *_accuracy_comparison.png, *_best_model.joblib
|                          # Per-dataset or older generated outputs
|-- fileignored/
|   |-- leakage_check_report.csv
|   |-- leave_one_source_out_results.csv
|   |-- social_media_buying_model_comparison.csv
|   |-- social_media_repeated_split_results.csv
|   |-- source_indicator_ablation.csv
|   `-- target_distribution.csv
|-- .venv/                 # Local virtual environment (not committed)
`-- __pycache__/           # Local Python cache (not committed)
```

Only `train_models.py`, `requirements.txt`, `README.md`, and `.gitignore` are
intended for version control. Respondent-level datasets, generated reports,
saved models, handoff copies, virtual environments, and Python caches are
excluded by `.gitignore`.

## Dataset and target rules

The script validates and combines three data sources:

- **Google Forms:** `Yes` becomes `0`; `No` and `I was considering buying
  something similar` become `1`.
- **Social Media Buying:** existing validated `impulse_purchase` values of
  `0` or `1` are used.
- **Psychological Drivers:** all `IB1` through `IB4` answers must be present.
  Their mean creates `impulse_target = 1` when at least `4`, otherwise `0`.

The Social Media labels already exist in the source file. The project does not
include the questionnaire, codebook, or script that originally produced them,
so their provenance and generalisation to future data cannot be established.

The script looks for datasets in `results/` first, then falls back to the
project root beside `train_models.py`. Generated reports and models are written
to `results/`. After a successful run, six selected handoff CSVs are copied to
`fileignored/`.

## Setup

1. Open the project folder in VS Code.
2. Create or activate a Python environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Obtain the original datasets through an authorised, private channel and place
   them in `results/`:

   - `Untitled form.csv`
   - `social_media_buying.csv`
   - `Cleaned dataset.xlsx`

   Do not commit or publicly upload respondent-level data.

5. Run the analysis:

   ```bash
   python train_models.py
   ```

## Modelling workflow

The script compares Logistic Regression, Random Forest, Gradient Boosting,
Extra Trees, and a Soft Voting Classifier. It uses an 80:20 stratified split
and five-fold stratified cross-validation. Preprocessing is fitted inside each
training pipeline, and model selection uses cross-validation accuracy rather
than test accuracy.

Additional checks include a source-indicator ablation, Social Media predictor
association analysis, leakage sensitivity, and repeated stratified splits.

The datasets do not provide a sufficiently comparable behavioural feature set.
The script therefore documents why common-feature and leave-one-source-out
experiments could not produce comparable results. The mixed-source experiment
uses valid source-specific features and must not be interpreted as proof of
generalisation to a new source.

For inference, provide the 48 pipeline feature columns, including
`source_dataset`. Leave unavailable source-specific fields empty or missing;
the pipeline imputes those values.

## Key findings

The audited run selected a Soft Voting Classifier consisting of Logistic
Regression, Random Forest, and Gradient Boosting. Its held-out accuracy was
79.58% (191 of 240 records correct). The Social Media subset was 87/87 correct,
while standalone repeated stratified splits ranged from 94% to 99%. These
results do not establish how the Social Media target labels were created or
performance on future data.

Removing the four most target-associated Social Media features together reduced
the seed-42 sensitivity split accuracy to 59%. This is an exploratory ablation
because those features were identified using target association, not a separate
independent estimate. The Psychological Drivers subset had an F1-score of
51.35%.

See `results/final_model_report.txt` for confusion matrices, feature
associations, split details, and limitations.

## Generated and legacy files

The main unified outputs are listed in the file structure above. The results
folder may also contain older per-dataset outputs such as
`ALL_MODEL_RESULTS.csv`, `*_model_comparison.csv`, plots, and saved models.
The current unified script does not generate or refresh those legacy files;
do not present them as results from the audited unified run.
