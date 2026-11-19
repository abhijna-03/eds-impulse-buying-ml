from pathlib import Path
import os
import shutil
import warnings

os.environ["MPLBACKEND"] = "Agg"

import joblib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
HANDOFF_DIR = BASE / "fileignored"
DATA_FILES = (
    "Untitled form.csv",
    "social_media_buying.csv",
    "Cleaned dataset.xlsx",
)
HANDOFF_FILES = [
    "social_media_buying_model_comparison.csv",
    "social_media_repeated_split_results.csv",
    "source_indicator_ablation.csv",
    "target_distribution.csv",
    "leave_one_source_out_results.csv",
    "leakage_check_report.csv",
]
RANDOM_STATE = 42
TARGET = "impulse_target"
SURVEY_NAME = "google_forms_survey"
SOCIAL_NAME = "social_media_buying"
PSYCH_NAME = "psychological_drivers"
SOURCES = (SURVEY_NAME, SOCIAL_NAME, PSYCH_NAME)
SURVEY_TARGET = (
    "Before seeing the product/offer, had you already planned to buy it?"
)
SURVEY_FEATURES = {
    "What is your age group?": "survey_age_group",
    "How often do you make purchases?": "survey_purchase_frequency",
    "Where did you make your most recent purchase?": "survey_channel",
    "Approximately how much did you spend?": "survey_spend_amount",
    "Was the product discounted?": "survey_discount_level",
    (
        "How much did social media, advertising, discounts, mood, peer "
        "opinion influence your decision?"
    ): "survey_influence_score",
    "How much did reviews and recommendations influence you?": (
        "survey_review_influence_score"
    ),
}
PSYCH_ALLOWED_FEATURES = [
    "GENDER",
    "AGEGRP",
    "EDU",
    "INCOME",
    "PLATFORM",
    "BUYFREQ",
    "PRODGRP",
    "PE1",
    "PE2",
    "PE3",
    "PE4",
    "SS1",
    "SS2",
    "SS3",
    "SS4",
    "SI1",
    "SI2",
    "SI3",
    "SI4",
    "PSV1",
    "PSV2",
    "PSV3",
    "PSV4",
    "FOMO1",
    "FOMO2",
    "FOMO3",
]
PSYCH_TARGET_ITEMS = ["IB1", "IB2", "IB3", "IB4"]
PSYCH_POST_PURCHASE_ITEMS = ["IG1", "IG2", "IG3", "IG4"]
SOCIAL_SUSPICIOUS_FOR_ABLATION = [
    "emotional_trigger_score",
    "ad_frequency",
    "peer_influence_score",
    "social_media_ads_clicks",
]
REPEATED_SPLIT_SEEDS = [1, 7, 21, 42, 84, 100, 123, 2024, 2025, 2026]
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def _normalize_headers(df):
    result = df.copy()
    result.columns = [" ".join(str(column).split()) for column in result.columns]
    return result


def _source_frames():
    input_directory = next(
        (
            directory
            for directory in (OUT, BASE)
            if all((directory / filename).is_file() for filename in DATA_FILES)
        ),
        None,
    )
    if input_directory is None:
        expected_paths = [
            str(directory / filename)
            for directory in (OUT, BASE)
            for filename in DATA_FILES
        ]
        raise FileNotFoundError(
            "Could not find all three input datasets together. Place them in "
            f"{OUT} (recommended) or beside the script. Checked: "
            + ", ".join(expected_paths)
        )

    survey = _normalize_headers(pd.read_csv(input_directory / DATA_FILES[0]))
    social = pd.read_csv(input_directory / DATA_FILES[1])
    psych = pd.read_excel(
        input_directory / DATA_FILES[2], sheet_name="Cleaned data"
    )
    return {
        SURVEY_NAME: survey,
        SOCIAL_NAME: social,
        PSYCH_NAME: psych,
    }


def _copy_handoff_files():
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in HANDOFF_FILES:
        source = OUT / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Expected handoff result was not generated: {source}"
            )
        destination = HANDOFF_DIR / filename
        if source.resolve() == destination.resolve():
            raise ValueError(f"Handoff destination must differ from source: {source}")
        shutil.copy2(source, destination)
        if source.stat().st_size != destination.stat().st_size:
            raise OSError(
                f"Copied handoff file failed size verification: {destination}"
            )
        copied.append(destination)
    return copied


def _coerce_target(raw_target, source):
    if source == SURVEY_NAME:
        normalized = raw_target.astype("string").str.strip()
        mapping = {
            "Yes": 0,
            "No": 1,
            "I was considering buying something similar": 1,
        }
        return normalized.map(mapping), "Unknown or missing survey response"
    if source == SOCIAL_NAME:
        values = pd.to_numeric(raw_target, errors="coerce")
        invalid = values.notna() & ~values.isin([0, 1])
        return values.mask(invalid), "Invalid or missing impulse_purchase"
    raise ValueError(f"Target coercion is not supported for source {source!r}")


def _prepare_sources(frames):
    prepared = {}
    audit_rows = []
    for source, frame in frames.items():
        if source == SURVEY_NAME:
            if SURVEY_TARGET not in frame.columns:
                raise ValueError(f"Survey target column not found: {SURVEY_TARGET}")
            missing = set(SURVEY_FEATURES).difference(frame.columns)
            if missing:
                raise ValueError(
                    f"Survey is missing required feature columns: {sorted(missing)}"
                )
            target, invalid_reason = _coerce_target(frame[SURVEY_TARGET], source)
            features = frame[list(SURVEY_FEATURES)].rename(columns=SURVEY_FEATURES)
            features["age_group"] = features["survey_age_group"]
            prepared[source] = (features, target)
            target_source_columns = [SURVEY_TARGET]
        elif source == SOCIAL_NAME:
            if "impulse_purchase" not in frame.columns:
                raise ValueError("Social dataset is missing impulse_purchase.")
            target, invalid_reason = _coerce_target(
                frame["impulse_purchase"], source
            )
            predictors = frame.drop(columns=["impulse_purchase"]).copy()
            predictors["age_group"] = pd.cut(
                pd.to_numeric(predictors["age"], errors="coerce"),
                bins=[-np.inf, 17, 20, 23, np.inf],
                labels=["17 or below", "18-20", "21-23", "24 or above"],
            ).astype("object")
            prepared[source] = (predictors, target)
            target_source_columns = ["impulse_purchase"]
        elif source == PSYCH_NAME:
            required = set(PSYCH_TARGET_ITEMS + PSYCH_POST_PURCHASE_ITEMS)
            missing = required.union(PSYCH_ALLOWED_FEATURES).difference(
                frame.columns
            )
            if missing:
                raise ValueError(
                    f"Psychological dataset is missing required columns: "
                    f"{sorted(missing)}"
                )
            items = frame[PSYCH_TARGET_ITEMS].apply(pd.to_numeric, errors="coerce")
            complete = items.notna().all(axis=1)
            mean_score = items.mean(axis=1)
            target = pd.Series(np.nan, index=frame.index, dtype="float64")
            target.loc[complete] = (mean_score.loc[complete] >= 4).astype(int)
            predictors = frame[PSYCH_ALLOWED_FEATURES].copy()
            # Demographic values are undocumented category codes, not measured
            # numeric quantities. Keep them categorical until a codebook exists.
            for column in ("GENDER", "AGEGRP", "EDU", "INCOME", "PLATFORM", "PRODGRP"):
                predictors[column] = (
                    predictors[column]
                    .astype("string")
                    .astype(object)
                    .where(predictors[column].notna(), np.nan)
                )
            prepared[source] = (predictors, target)
            invalid_reason = "Incomplete or invalid IB1-IB4 target items"
            target_source_columns = PSYCH_TARGET_ITEMS
        else:
            raise ValueError(f"Unexpected dataset source: {source}")

        invalid_count = int(target.isna().sum())
        audit_rows.append(
            {
                "source_dataset": source,
                "input_rows": len(frame),
                "target_columns": "; ".join(target_source_columns),
                "valid_target_rows": int(target.notna().sum()),
                "removed_missing_or_invalid_target_rows": invalid_count,
                "removal_reason": invalid_reason if invalid_count else "",
                "valid_class_0": int((target == 0).sum()),
                "valid_class_1": int((target == 1).sum()),
            }
        )
    return prepared, pd.DataFrame(audit_rows)


def _build_extended_dataset(prepared):
    X_parts = []
    y_parts = []
    for source in SOURCES:
        features, target = prepared[source]
        valid = target.notna()
        source_X = features.loc[valid].copy()
        source_X["source_dataset"] = source
        X_parts.append(source_X.reset_index(drop=True))
        y_parts.append(target.loc[valid].astype(int).reset_index(drop=True))
    X = pd.concat(X_parts, ignore_index=True, sort=False)
    y = pd.concat(y_parts, ignore_index=True).rename(TARGET)
    if not y.isin([0, 1]).all():
        raise ValueError("Unified target contains values other than 0 and 1.")
    return X, y


def _target_distribution(prepared, y):
    rows = []
    for source in SOURCES:
        _, target = prepared[source]
        valid = target.dropna().astype(int)
        rows.append(
            {
                "source_dataset": source,
                "class_0_count": int((valid == 0).sum()),
                "class_1_count": int((valid == 1).sum()),
                "valid_target_rows": len(valid),
                "removed_missing_or_invalid_target_rows": int(target.isna().sum()),
            }
        )
    rows.append(
        {
            "source_dataset": "combined",
            "class_0_count": int((y == 0).sum()),
            "class_1_count": int((y == 1).sum()),
            "valid_target_rows": len(y),
            "removed_missing_or_invalid_target_rows": sum(
                int(target.isna().sum()) for _, target in prepared.values()
            ),
        }
    )
    return pd.DataFrame(rows)


def _dataset_summary(frames):
    rows = []
    for source, frame in frames.items():
        for column in frame.columns:
            values = frame[column]
            row = {
                "source_dataset": source,
                "rows": len(frame),
                "column": column,
                "dtype": str(values.dtype),
                "missing_count": int(values.isna().sum()),
                "unique_nonmissing_values": int(values.nunique(dropna=True)),
                "duplicate_full_rows_in_dataset": int(frame.duplicated().sum()),
            }
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                row.update(
                    {
                        "mean": numeric.mean(),
                        "std": numeric.std(),
                        "min": numeric.min(),
                        "q25": numeric.quantile(0.25),
                        "median": numeric.median(),
                        "q75": numeric.quantile(0.75),
                        "max": numeric.max(),
                        "top_values": "",
                    }
                )
            else:
                top = values.value_counts(dropna=False).head(20)
                row.update(
                    {
                        "mean": np.nan,
                        "std": np.nan,
                        "min": "",
                        "q25": np.nan,
                        "median": np.nan,
                        "q75": np.nan,
                        "max": "",
                        "top_values": " | ".join(
                            f"{value!r}: {count}" for value, count in top.items()
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_exclusion_report():
    rows = [
        {
            "source_dataset": SURVEY_NAME,
            "feature": "Timestamp",
            "decision": "excluded",
            "reason": "Response timestamp is not an intended behavioral predictor.",
        },
        {
            "source_dataset": SURVEY_NAME,
            "feature": SURVEY_TARGET,
            "decision": "excluded",
            "reason": "This response defines impulse_target.",
        },
        {
            "source_dataset": SURVEY_NAME,
            "feature": (
                "After buying the product, how much did you feel it was "
                "unnecessary?"
            ),
            "decision": "excluded",
            "reason": "Post-purchase assessment is unavailable at prediction time.",
        },
        {
            "source_dataset": SOCIAL_NAME,
            "feature": "impulse_purchase",
            "decision": "excluded",
            "reason": "This column defines impulse_target.",
        },
        {
            "source_dataset": SOCIAL_NAME,
            "feature": "All remaining supplied predictors",
            "decision": "retained in extended experiment; ablated in sensitivity check",
            "reason": (
                "No predictor is an exact binary target copy in observed data. "
                "Timing and data-generation provenance are unavailable; "
                "sensitivity analysis separately excludes the four most "
                "target-associated ad/influence measures."
            ),
        },
    ]
    for feature in PSYCH_TARGET_ITEMS:
        rows.append(
            {
                "source_dataset": PSYCH_NAME,
                "feature": feature,
                "decision": "excluded",
                "reason": "Item is used to calculate impulse_target.",
            }
        )
    for feature in PSYCH_POST_PURCHASE_ITEMS:
        rows.append(
            {
                "source_dataset": PSYCH_NAME,
                "feature": feature,
                "decision": "excluded",
                "reason": (
                    "IG items measure gratification after an impulsive purchase "
                    "and are not appropriate for prospective prediction."
                ),
            }
        )
    rows.append(
        {
            "source_dataset": PSYCH_NAME,
            "feature": "Other documented requested predictors",
            "decision": "retained in extended experiment",
            "reason": (
                "Allowed demographic and construct items are retained; coded "
                "demographic fields are categorical because no codebook was found."
            ),
        }
    )
    return pd.DataFrame(rows)


def _social_leakage_report(social):
    y = pd.to_numeric(social["impulse_purchase"], errors="coerce")
    predictors = social.drop(columns=["impulse_purchase"])
    duplicate_predictor_rows = int(predictors.duplicated().sum())
    conflicting_duplicate_groups = int(
        social.groupby(list(predictors.columns), dropna=False)["impulse_purchase"]
        .nunique()
        .gt(1)
        .sum()
    )
    rows = []
    for column in predictors.columns:
        values = predictors[column]
        numeric = pd.to_numeric(values, errors="coerce")
        is_numeric = numeric.notna().sum() > 0
        if is_numeric:
            association = numeric.corr(y)
        else:
            association = np.nan

        if is_numeric and numeric.nunique(dropna=True) > 20:
            grouping = pd.qcut(numeric, q=10, duplicates="drop")
            purity_method = "numeric deciles"
        else:
            grouping = values
            purity_method = "observed category/value"
        group_table = pd.DataFrame(
            {"group": grouping, "target": y}
        ).dropna(subset=["target"])
        grouped = group_table.groupby("group", observed=True)["target"].agg(
            ["count", "mean"]
        )
        if grouped.empty:
            maximum_purity = np.nan
            maximum_group_rows = 0
            maximum_purity_group = ""
        else:
            grouped["purity"] = grouped[["mean"]].max(axis=1)
            grouped["purity"] = grouped["purity"].where(
                grouped["purity"] >= 0.5, 1 - grouped["purity"]
            )
            grouped = grouped.sort_values(
                ["purity", "count"], ascending=[False, False]
            )
            highest_group = grouped.iloc[0]
            maximum_purity = float(highest_group["purity"])
            maximum_group_rows = int(highest_group["count"])
            maximum_purity_group = str(grouped.index[0])
        exact_binary_copy = False
        if numeric.notna().all() and numeric.nunique() <= 2:
            exact_binary_copy = bool(
                np.array_equal(numeric.to_numpy(), y.to_numpy())
                or np.array_equal(1 - numeric.to_numpy(), y.to_numpy())
            )
        if exact_binary_copy:
            classification = "exact target duplicate"
        elif maximum_purity == 1 and maximum_group_rows >= 5:
            classification = "perfect target separation for a group with >=5 rows"
        elif maximum_purity >= 0.95 and maximum_group_rows >= 5:
            classification = "near-perfect target separation for a group with >=5 rows"
        else:
            classification = "no high-support perfect relation observed; provenance/timing unknown"
        rows.append(
            {
                "source_dataset": SOCIAL_NAME,
                "predictor": column,
                "dtype": str(values.dtype),
                "unique_values": int(values.nunique(dropna=True)),
                "target_correlation_or_binary_association": association,
                "maximum_within_value_target_class_purity": maximum_purity,
                "purity_grouping": purity_method,
                "highest_purity_group": maximum_purity_group,
                "rows_in_highest_purity_group": maximum_group_rows,
                "exact_binary_target_copy": exact_binary_copy,
                "classification": classification,
                "duplicate_predictor_rows": duplicate_predictor_rows,
                "duplicate_predictor_groups_with_conflicting_target": (
                    conflicting_duplicate_groups
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_test_overlap(X_train, X_test, y_train, y_test):
    train_rows = X_train.copy()
    train_rows[TARGET] = y_train.to_numpy()
    test_rows = X_test.copy()
    test_rows[TARGET] = y_test.to_numpy()
    train_hashes = pd.util.hash_pandas_object(train_rows, index=False)
    test_hashes = pd.util.hash_pandas_object(test_rows, index=False)
    return int(test_hashes.isin(set(train_hashes)).sum())


def build_pipeline(X, estimator):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("classifier", estimator)]
    )


def _candidate_estimators():
    logistic = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return {
        "Logistic Regression": logistic,
        "Random Forest": forest,
        "Gradient Boosting": boosting,
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Soft Voting Classifier": VotingClassifier(
            estimators=[
                ("logistic", logistic),
                ("random_forest", forest),
                ("gradient_boosting", boosting),
            ],
            voting="soft",
            n_jobs=1,
        ),
    }


def _metric_values(y_true, prediction, probability):
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Balanced accuracy": balanced_accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1-score": f1_score(y_true, prediction, zero_division=0),
        "ROC-AUC": (
            roc_auc_score(y_true, probability)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
    }


def _cv_result_row(cv_result, prefix):
    row = {}
    for metric in SCORING:
        values = cv_result[f"test_{metric}"]
        row[f"{prefix} {metric.replace('_', ' ')} mean"] = float(
            np.mean(values)
        )
        row[f"{prefix} {metric.replace('_', ' ')} std"] = float(
            np.std(values, ddof=0)
        )
    return row


def _evaluate_candidates(X_train, y_train, X_test, y_test, prefix="CV"):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    metrics_by_model = {}
    for name, estimator in _candidate_estimators().items():
        model = build_pipeline(X_train, estimator)
        cv_result = cross_validate(
            model,
            X_train,
            y_train,
            cv=folds,
            scoring=SCORING,
            error_score="raise",
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        metrics = _metric_values(y_test, prediction, probability)
        row = {"Model": name}
        row.update(_cv_result_row(cv_result, prefix))
        row.update({f"Test {key}": value for key, value in metrics.items()})
        rows.append(row)
        fitted[name] = model
        metrics_by_model[name] = metrics
        print(
            f"{name}: CV accuracy "
            f"{row[f'{prefix} accuracy mean']:.3f} "
            f"(+/- {row[f'{prefix} accuracy std']:.3f}); "
            f"test accuracy {metrics['Accuracy']:.3f}"
        )
    return pd.DataFrame(rows), fitted, metrics_by_model


def _run_extended_experiment(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    overlap = _source_test_overlap(X_train, X_test, y_train, y_test)
    if overlap:
        raise RuntimeError(f"Found {overlap} train/test record overlaps.")
    comparison, fitted, _ = _evaluate_candidates(
        X_train, y_train, X_test, y_test
    )
    comparison["source_indicator_included"] = True
    selected_index = comparison["CV accuracy mean"].idxmax()
    selected_name = comparison.loc[selected_index, "Model"]
    selected_model = fitted[selected_name]
    selected_prediction = selected_model.predict(X_test)
    selected_probability = selected_model.predict_proba(X_test)[:, 1]
    selected_metrics = _metric_values(
        y_test, selected_prediction, selected_probability
    )

    source_test = X_test["source_dataset"].copy()
    source_rows = []
    test_detail = pd.DataFrame(
        {
            "source_dataset": source_test.to_numpy(),
            "target": y_test.to_numpy(),
            "prediction": selected_prediction,
            "probability": selected_probability,
        },
        index=y_test.index,
    )
    for source in SOURCES:
        subset = test_detail.loc[test_detail["source_dataset"] == source]
        if subset.empty:
            source_rows.append(
                {
                    "source_dataset": source,
                    "test_rows": 0,
                    "class_0_count": 0,
                    "class_1_count": 0,
                    "Accuracy": np.nan,
                    "Balanced accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-score": np.nan,
                    "ROC-AUC": np.nan,
                }
            )
            continue
        source_metrics = _metric_values(
            subset["target"], subset["prediction"], subset["probability"]
        )
        source_rows.append(
            {
                "source_dataset": source,
                "test_rows": len(subset),
                "class_0_count": int((subset["target"] == 0).sum()),
                "class_1_count": int((subset["target"] == 1).sum()),
                **source_metrics,
            }
        )
    source_performance = pd.DataFrame(source_rows)

    no_source_X_train = X_train.drop(columns=["source_dataset"])
    no_source_X_test = X_test.drop(columns=["source_dataset"])
    no_source_rows = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, estimator in _candidate_estimators().items():
        cv_model = build_pipeline(no_source_X_train, estimator)
        cv_result = cross_validate(
            cv_model,
            no_source_X_train,
            y_train,
            cv=folds,
            scoring=SCORING,
            error_score="raise",
            n_jobs=1,
        )
        test_model = build_pipeline(no_source_X_train, estimator)
        test_model.fit(no_source_X_train, y_train)
        prediction = test_model.predict(no_source_X_test)
        probability = test_model.predict_proba(no_source_X_test)[:, 1]
        metrics = _metric_values(y_test, prediction, probability)
        row = {"Model": name, "source_indicator_included": False}
        row.update(_cv_result_row(cv_result, "CV"))
        row.update({f"Test {key}": value for key, value in metrics.items()})
        no_source_rows.append(row)
    ablation = pd.concat(
        [comparison, pd.DataFrame(no_source_rows)], ignore_index=True
    )
    full_cv = comparison.set_index("Model")["CV accuracy mean"]
    without_cv = pd.DataFrame(no_source_rows).set_index("Model")[
        "CV accuracy mean"
    ]
    ablation["paired_CV_accuracy_delta_vs_without_source"] = ablation.apply(
        lambda row: (
            full_cv[row["Model"]] - without_cv[row["Model"]]
            if row["Model"] in full_cv.index
            else np.nan
        ),
        axis=1,
    )

    # One panel per classifier, all sharing the same held-out split.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.flat, comparison["Model"]):
        ConfusionMatrixDisplay.from_estimator(
            fitted[name],
            X_test,
            y_test,
            display_labels=[0, 1],
            values_format="d",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    axes.flat[-1].axis("off")
    fig.suptitle("Extended-feature experiment: held-out confusion matrices")
    fig.tight_layout()
    fig.savefig(OUT / "unified_confusion_matrix.png", dpi=220)
    plt.close(fig)

    plot = comparison.sort_values("Test Accuracy")
    plt.figure(figsize=(9, 5))
    plt.barh(plot["Model"], plot["Test Accuracy"])
    plt.xlabel("Held-out test accuracy")
    plt.xlim(0, 1)
    plt.title("Extended-feature model accuracy comparison")
    plt.tight_layout()
    plt.savefig(OUT / "unified_accuracy_comparison.png", dpi=220)
    plt.close()

    return {
        "comparison": comparison,
        "ablation": ablation,
        "selected_name": selected_name,
        "selected_model": selected_model,
        "selected_metrics": selected_metrics,
        "source_performance": source_performance,
        "y_test": y_test,
        "test_detail": test_detail,
        "split_overlap": overlap,
    }


def _run_social_sensitivity(social, target):
    y = pd.to_numeric(target, errors="coerce")
    valid = y.notna()
    X = social.drop(columns=["impulse_purchase"]).loc[valid].copy()
    y = y.loc[valid].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    X["age_group"] = pd.cut(
        pd.to_numeric(X["age"], errors="coerce"),
        bins=[-np.inf, 17, 20, 23, np.inf],
        labels=["17 or below", "18-20", "21-23", "24 or above"],
    ).astype("object")

    full_train, full_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    class_counts = y_test.value_counts().to_dict()
    if _source_test_overlap(full_train, full_test, y_train, y_test):
        raise RuntimeError("Social-media train/test overlap detected.")

    feature_sets = {
        "all_predictors": X.columns.tolist(),
        "without_high_association_sensitivity_features": [
            column
            for column in X.columns
            if column not in SOCIAL_SUSPICIOUS_FOR_ABLATION
        ],
    }
    rows = []
    selected_by_set = {}
    for feature_set, columns in feature_sets.items():
        train = full_train[columns]
        test = full_test[columns]
        results, fitted, _ = _evaluate_candidates(
            train, y_train, test, y_test, prefix="CV"
        )
        for _, result_row in results.iterrows():
            rows.append(
                {
                    "feature_set": feature_set,
                    "excluded_features": (
                        "; ".join(SOCIAL_SUSPICIOUS_FOR_ABLATION)
                        if feature_set != "all_predictors"
                        else ""
                    ),
                    **result_row.to_dict(),
                }
            )
        best_idx = results["CV accuracy mean"].idxmax()
        best_name = results.loc[best_idx, "Model"]
        selected_by_set[feature_set] = best_name

    result_table = pd.DataFrame(rows)
    full_results = result_table.loc[
        result_table["feature_set"] == "all_predictors"
    ].set_index("Model")
    no_suspicious_results = result_table.loc[
        result_table["feature_set"]
        == "without_high_association_sensitivity_features"
    ].set_index("Model")
    result_table["test_accuracy_delta_vs_all_predictors"] = result_table.apply(
        lambda row: (
            row["Test Accuracy"] - full_results.loc[row["Model"], "Test Accuracy"]
            if row["Model"] in full_results.index
            else np.nan
        ),
        axis=1,
    )
    selected_name = selected_by_set["all_predictors"]
    selected_model_row = result_table.loc[
        (result_table["feature_set"] == "all_predictors")
        & (result_table["Model"] == selected_name)
    ].iloc[0]

    repeat_rows = []
    selected_estimator = _candidate_estimators()[selected_name]
    for seed in REPEATED_SPLIT_SEEDS:
        train_indices, test_indices = train_test_split(
            np.arange(len(X)),
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )
        train_X, test_X = X.iloc[train_indices], X.iloc[test_indices]
        train_y, test_y = y.iloc[train_indices], y.iloc[test_indices]
        if _source_test_overlap(train_X, test_X, train_y, test_y):
            raise RuntimeError(
                f"Social-media overlap detected for repeated split seed {seed}."
            )
        model = build_pipeline(train_X, selected_estimator)
        model.fit(train_X, train_y)
        prediction = model.predict(test_X)
        probability = model.predict_proba(test_X)[:, 1]
        repeat_rows.append(
            {
                "seed": seed,
                "test_rows": len(test_y),
                "class_0_count": int((test_y == 0).sum()),
                "class_1_count": int((test_y == 1).sum()),
                **_metric_values(test_y, prediction, probability),
            }
        )
    repeat_table = pd.DataFrame(repeat_rows)
    audit = {
        "records": len(X),
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
        "test_rows_seed_42": len(y_test),
        "test_class_0_seed_42": int(class_counts.get(0, 0)),
        "test_class_1_seed_42": int(class_counts.get(1, 0)),
        "selected_model_by_cv": selected_name,
        "selected_model_seed42_test_accuracy": float(
            selected_model_row["Test Accuracy"]
        ),
        "repeated_accuracy_min": float(repeat_table["Accuracy"].min()),
        "repeated_accuracy_max": float(repeat_table["Accuracy"].max()),
        "repeated_accuracy_mean": float(repeat_table["Accuracy"].mean()),
        "sensitivity_selected_model": selected_by_set[
            "without_high_association_sensitivity_features"
        ],
        "sensitivity_test_accuracy": float(
            no_suspicious_results.loc[
                selected_by_set[
                    "without_high_association_sensitivity_features"
                ],
                "Test Accuracy",
            ]
        ),
    }
    return result_table, repeat_table, audit


def _harmonization_report():
    return (
        "A common-feature experiment was not run. The available source files do "
        "not define a fully comparable behavioral predictor set across all three "
        "questionnaires. Google Forms age group uses readable age bands and "
        "social_media_buying contains age in years (these two can be aligned), "
        "but the psychological workbook uses numeric AGEGRP codes with no "
        "codebook. BUYFREQ is also an undocumented category code, while the "
        "social dataset's online_purchase_frequency is a count with no stated "
        "time window; the survey uses verbal frequency categories. Retail "
        "channel, social platform, and coded psychological PLATFORM are not "
        "interchangeable. Survey discount level, perceived_discount, and the "
        "psychological questionnaire do not have a documented common scale. "
        "Mapping any of these across all three sources would add unsupported "
        "assumptions. The extended-feature experiment retains valid source "
        "specific measures with missing values for other sources and uses "
        "training-fold imputation. This estimates performance on a mixture of "
        "these datasets, not guaranteed cross-source generalization."
    )


def _write_common_feature_outputs():
    reason = _harmonization_report()
    pd.DataFrame(
        [
            {
                "experiment": "A - common features",
                "status": "not performed",
                "reason": reason,
            }
        ]
    ).to_csv(OUT / "common_feature_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "training_sources": "",
                "test_source": source,
                "status": "not performed",
                "reason": (
                    "Leave-one-source-out evaluation requires valid, comparable "
                    "predictor definitions. No such three-source feature set "
                    "can be established from the available codebooks/data."
                ),
            }
            for source in SOURCES
        ]
    ).to_csv(OUT / "leave_one_source_out_results.csv", index=False)


def _write_final_report(
    frames,
    target_audit,
    y,
    X,
    experiment,
    leakage_report,
    social_audit,
    repeated_splits,
):
    comparison = experiment["comparison"].sort_values(
        "CV accuracy mean", ascending=False
    )
    by_source = experiment["source_performance"]
    metrics = experiment["selected_metrics"]
    source_indicator_deltas = experiment["ablation"].loc[
        experiment["ablation"]["source_indicator_included"]
    ].set_index("Model")["paired_CV_accuracy_delta_vs_without_source"]
    perfect_groups = leakage_report.loc[
        leakage_report["classification"].str.startswith(
            "perfect target separation"
        )
    ]
    lines = [
        "Impulse Buying Behaviour - Final Model and Leakage Audit",
        "=" * 66,
        "",
        "Dataset accounting",
        f"  Survey source rows: {len(frames[SURVEY_NAME])}",
        f"  Social source rows: {len(frames[SOCIAL_NAME])}",
        f"  Psychological source rows: {len(frames[PSYCH_NAME])}",
        f"  Total rows before target validation: {sum(map(len, frames.values()))}",
        f"  Rows retained with valid target: {len(y)}",
        f"  Rows removed for missing/invalid target: "
        f"{int(target_audit['removed_missing_or_invalid_target_rows'].sum())}",
        f"  Features in extended dataset: {X.shape[1]} (including source_dataset)",
        "",
        "Target definitions",
        "  Google Forms: Yes=0; No=1; considering a similar purchase=1.",
        "  Social Media: impulse_purchase must be 0 or 1.",
        "  Psychological Drivers: all IB1-IB4 required; mean >= 4 gives 1, "
        "otherwise 0.",
        "",
        "Feature validity and harmonization",
        "  Excluded survey Timestamp, the survey target, and the post-purchase "
        "unnecessary-purchase item.",
        "  Excluded social impulse_purchase target from predictors. No observed "
        "social predictor was an exact binary target copy.",
        "  Excluded psychological IB1-IB4 (target construction) and IG1-IG4 "
        "(post-purchase gratification).",
        "  Psychological demographic codes lack a codebook and were treated as "
        "categorical, not continuous.",
        "  Extended source-specific features retained:",
    ]
    for source in SOURCES:
        source_X = X.loc[X["source_dataset"] == source].drop(
            columns=["source_dataset"]
        )
        active = source_X.columns[source_X.notna().any()].tolist()
        lines.append(f"    {source}: {', '.join(active)}")
    lines.extend(
        [
            f"  Common-feature experiment: not run. {_harmonization_report()}",
            "",
            "Model selection and validation",
            "  One stratified 80:20 split, random_state=42. All preprocessing is "
            "inside a Pipeline and fitted only on each training fold/split.",
            "  Five-fold StratifiedKFold on training rows only. Selected by mean "
            "cross-validation accuracy only; test scores did not select the model.",
            f"  Selected model: {experiment['selected_name']}",
            "",
            "Five-model results (cross-validation accuracy mean +/- std):",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"  {row['Model']}: {row['CV accuracy mean']:.4f} +/- "
            f"{row['CV accuracy std']:.4f}; held-out accuracy "
            f"{row['Test Accuracy']:.4f}"
        )
    lines.extend(
        [
            "",
            "Selected model held-out metrics",
            *[
                f"  {name}: {value:.4f}"
                for name, value in metrics.items()
            ],
            "",
            "Selected model test performance by source (separate source "
            "subgroups from the same random test split):",
        ]
    )
    for _, row in by_source.iterrows():
        lines.append(
            f"  {row['source_dataset']}: n={int(row['test_rows'])}, "
            f"class 0={int(row['class_0_count'])}, "
            f"class 1={int(row['class_1_count'])}, "
            f"accuracy={row['Accuracy']:.4f}, "
            f"balanced accuracy={row['Balanced accuracy']:.4f}, "
            f"F1={row['F1-score']:.4f}"
        )
    lines.extend(
        [
            "",
            "Source indicator and Social Media accuracy investigation",
            "  Paired CV accuracy changes from including source_dataset were "
            + "; ".join(
                f"{model} {delta:+.4f}"
                for model, delta in source_indicator_deltas.items()
            )
            + ". The changes are small in this run (largest absolute change "
            f"{source_indicator_deltas.abs().max():.4f}), so source identity "
            "alone did not materially change aggregate CV accuracy.",
            f"  Social rows={social_audit['records']}, class 0="
            f"{social_audit['class_0']}, class 1={social_audit['class_1']}.",
            f"  Social seed-42 test rows={social_audit['test_rows_seed_42']}, "
            f"class 0={social_audit['test_class_0_seed_42']}, class "
            f"1={social_audit['test_class_1_seed_42']}.",
            f"  Standalone Social Media model selected by CV: "
            f"{social_audit['selected_model_by_cv']}; test accuracy="
            f"{social_audit['selected_model_seed42_test_accuracy']:.4f}.",
            f"  Across {len(repeated_splits)} fixed-model, repeated stratified "
            f"80:20 splits, accuracy range="
            f"{social_audit['repeated_accuracy_min']:.4f}-"
            f"{social_audit['repeated_accuracy_max']:.4f} "
            f"(mean={social_audit['repeated_accuracy_mean']:.4f}).",
            f"  After removing the four highest association sensitivity "
            f"features, selected algorithm="
            f"{social_audit['sensitivity_selected_model']}; its test accuracy="
            f"{social_audit['sensitivity_test_accuracy']:.4f}.",
            "  Sensitivity features are not declared leaked: they were chosen "
            "for ablation because they showed the strongest individual "
            "associations and because their measurement timing is undocumented.",
            "  Perfect small-group associations found by the Social Media "
            "screen: "
            + (
                "; ".join(
                    f"{row['predictor']} group {row['highest_purity_group']} "
                    f"(n={int(row['rows_in_highest_purity_group'])})"
                    for _, row in perfect_groups.iterrows()
                )
                if not perfect_groups.empty
                else "none"
            )
            + ". These are association flags, not proof of target encoding; "
            "the strongest reported group has only five records.",
            "  No exact duplicates or conflicting duplicate predictor groups "
            "were observed in the Social Media data. A random split of unique "
            "rows has no exact train/test row overlap. Pipeline preprocessing "
            "is fitted within each training fold.",
            "  No questionnaire, codebook, or data-generation script was "
            "available to establish when social predictors were measured or "
            "whether the target was generated from them. Therefore, 100% "
            "accuracy on any one split is not proof of leakage or of genuine "
            "generalization; inspect the leakage_check_report.csv associations "
            "and repeated-split results.",
            "",
            "Limitations",
            "  The target definitions differ across sources and are not "
            "guaranteed to measure exactly the same construct.",
            "  Most source-specific columns are structurally missing outside "
            "their originating dataset; a model may learn source-specific "
            "patterns. Compare source_indicator_ablation.csv and "
            "unified_source_test_performance.csv.",
            "  Small survey subgroup test counts make its metrics noisy.",
            "  Cross-source generalization was not evaluated because "
            "comparable predictors cannot be defined from the available "
            "questionnaire documentation.",
            "  Results are predictive associations, not causal evidence.",
            "",
            "Target counts by source:",
        ]
    )
    for _, row in target_audit.iterrows():
        lines.append(
            f"  {row['source_dataset']}: {int(row['valid_target_rows'])} valid; "
            f"{int(row['removed_missing_or_invalid_target_rows'])} removed; "
            f"class 0={int(row['valid_class_0'])}, "
            f"class 1={int(row['valid_class_1'])}"
        )
    (OUT / "final_model_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    warnings.filterwarnings(
        "ignore",
        message="The least populated class in y has only",
        category=UserWarning,
    )
    OUT.mkdir(exist_ok=True)
    frames = _source_frames()
    prepared, target_audit = _prepare_sources(frames)
    X, y = _build_extended_dataset(prepared)

    expected_rows = {SURVEY_NAME: 108, SOCIAL_NAME: 500, PSYCH_NAME: 589}
    for source, expected in expected_rows.items():
        actual = len(frames[source])
        if actual != expected:
            raise ValueError(
                f"{source} has {actual} rows; expected {expected}. "
                "Inspect the input data before training."
            )
    if len(X) != sum(expected_rows.values()):
        raise ValueError(
            f"Expected 1197 valid target rows but found {len(X)}. "
            "See target audit for excluded records."
        )
    if y.nunique() != 2 or not y.isin([0, 1]).all():
        raise ValueError("Unified target must contain both binary classes 0 and 1.")

    _dataset_summary(frames).to_csv(OUT / "dataset_summary.csv", index=False)
    target_audit.to_csv(OUT / "target_distribution.csv", index=False)
    _feature_exclusion_report().to_csv(
        OUT / "feature_exclusion_report.csv", index=False
    )
    leakage_report = _social_leakage_report(frames[SOCIAL_NAME])
    leakage_report.to_csv(OUT / "leakage_check_report.csv", index=False)
    _write_common_feature_outputs()

    print("Source rows:", {source: len(frame) for source, frame in frames.items()})
    print("Unified target rows:", len(y))
    print("Target distribution:\n", y.value_counts().sort_index().to_string())
    print(
        "Removed target rows:",
        int(target_audit["removed_missing_or_invalid_target_rows"].sum()),
    )

    experiment = _run_extended_experiment(X, y)
    experiment["comparison"].to_csv(
        OUT / "unified_model_comparison.csv", index=False
    )
    experiment["ablation"].to_csv(
        OUT / "source_indicator_ablation.csv", index=False
    )
    experiment["source_performance"].to_csv(
        OUT / "unified_source_test_performance.csv", index=False
    )
    joblib.dump(
        experiment["selected_model"], OUT / "unified_best_model.joblib"
    )

    social_target, _ = _coerce_target(
        frames[SOCIAL_NAME]["impulse_purchase"], SOCIAL_NAME
    )
    social_results, repeated_splits, social_audit = _run_social_sensitivity(
        frames[SOCIAL_NAME], social_target
    )
    social_results.to_csv(OUT / "social_media_leakage_sensitivity.csv", index=False)
    social_model_comparison = social_results.loc[
        social_results["feature_set"] == "all_predictors"
    ].drop(
        columns=[
            "feature_set",
            "excluded_features",
            "test_accuracy_delta_vs_all_predictors",
        ]
    )
    social_model_comparison.to_csv(
        OUT / "social_media_buying_model_comparison.csv", index=False
    )
    repeated_splits.to_csv(
        OUT / "social_media_repeated_split_results.csv", index=False
    )
    _write_final_report(
        frames,
        target_audit,
        y,
        X,
        experiment,
        leakage_report,
        social_audit,
        repeated_splits,
    )

    loaded_model = joblib.load(OUT / "unified_best_model.joblib")
    sample_predictions = loaded_model.predict(X.iloc[:5])
    expected_outputs = [
        "unified_model_comparison.csv",
        "unified_source_test_performance.csv",
        "unified_confusion_matrix.png",
        "unified_accuracy_comparison.png",
        "unified_best_model.joblib",
        "target_distribution.csv",
        "feature_exclusion_report.csv",
        "leakage_check_report.csv",
        "dataset_summary.csv",
        "common_feature_results.csv",
        "extended_feature_results.csv",
        "leave_one_source_out_results.csv",
        "final_model_report.txt",
        "source_indicator_ablation.csv",
        "social_media_leakage_sensitivity.csv",
        "social_media_buying_model_comparison.csv",
        "social_media_repeated_split_results.csv",
    ]
    experiment["comparison"].to_csv(
        OUT / "extended_feature_results.csv", index=False
    )
    missing_outputs = [
        filename
        for filename in expected_outputs
        if not (OUT / filename).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Expected output files were not created: {missing_outputs}")

    copied_handoff_files = _copy_handoff_files()
    print("\nSelected extended model:", experiment["selected_name"])
    print("Held-out metrics:", experiment["selected_metrics"])
    print("\nSource test performance:\n", experiment["source_performance"].to_string(index=False))
    print("\nSocial Media audit:", social_audit)
    print("Loaded saved model and generated sample predictions:", sample_predictions)
    print("Copied handoff files:")
    for path in copied_handoff_files:
        print(f"  {path}")
    print(f"All requested outputs saved to {OUT}")


if __name__ == "__main__":
    main()
