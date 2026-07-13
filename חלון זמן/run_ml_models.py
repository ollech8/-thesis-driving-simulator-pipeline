"""
run_ml_models.py
----------------
Trains XGBoost and Random Forest on each window dataset (1s-5s).
Compares model performance across window sizes.

Design decisions:
  - Group split by participant (Id) — no participant appears in both train and test
  - Stratified by has_feedback (does participant have at least one feedback event)
  - Same train/test participant split applied to all 5 window datasets for fair comparison
  - Class imbalance: scale_pos_weight (XGBoost), class_weight='balanced' (RF)
  - One-hot encoding for all categorical columns
  - Identifier columns dropped before training
  - Default hyperparameters, tuning later
  - Results printed to console only
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\דיבאג\חלון זמן"
WINDOW_SIZES = [1, 2, 3, 4, 5]
TARGET_COL   = "feedback_next_1s"
RANDOM_STATE = 42

# Columns to drop before training (identifiers, not features)
DROP_COLS = [
    "Id", "event_id", "window_index",
    "start_time", "end_time", "window_size_sec",
    "is_partial_window"
]

# Categorical columns to one-hot encode
CAT_COLS = [
    "Condition", "Map",
    "relevant_object_name",
    "TrafficLight_prev", "TrafficLight_current",
    "prev_phase", "current_phase",
    "event_category"
]

# ─── Helper: print section header ────────────────────────────────────────────
def header(text):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")

# ─── Step 1: Determine train/test participant split ───────────────────────────
# Load the 1s dataset just to get participant-level feedback info
# The same split will be reused for all window sizes

header("Determining participant split (based on 1s dataset)")

df_ref = pd.read_csv(os.path.join(DATA_DIR, "windows_1s_clean.csv"))

# For each participant, check if they have at least one feedback event
participant_fb = (
    df_ref.groupby("Id")[TARGET_COL]
    .max()
    .reset_index()
    .rename(columns={TARGET_COL: "has_feedback"})
)

print(f"  Total participants: {len(participant_fb)}")
print(f"  With feedback:      {participant_fb['has_feedback'].sum()}")
print(f"  Without feedback:   {(participant_fb['has_feedback'] == 0).sum()}")

# Stratified split: maintain same has_feedback ratio in train/test
# Split within each stratum (has_feedback=0 and has_feedback=1 separately)
np.random.seed(RANDOM_STATE)

train_ids = []
test_ids  = []

for fb_val in [0, 1]:
    stratum = participant_fb[participant_fb["has_feedback"] == fb_val]["Id"].tolist()
    np.random.shuffle(stratum)
    n_test  = max(1, round(len(stratum) * 0.20))
    test_ids.extend(stratum[:n_test])
    train_ids.extend(stratum[n_test:])

train_ids = set(train_ids)
test_ids  = set(test_ids)

print(f"\n  Train participants: {len(train_ids)}")
print(f"  Train IDs: {sorted(train_ids)}")
print(f"\n  Test participants:  {len(test_ids)}")
print(f"  Test IDs:  {sorted(test_ids)}")

print(f"\n  {'Split':<8} {'Id':<15} {'has_feedback':<15} {'n_windows':<12} {'n_positive':<12} {'feedback_rate':>13}")
print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12} {'-'*12} {'-'*13}")

for split_label, id_set in [("TRAIN", train_ids), ("TEST", test_ids)]:
    split_df    = df_ref[df_ref["Id"].isin(id_set)]
    n_fb_events = (split_df.groupby("event_id")[TARGET_COL].sum() > 0).sum()
    for pid in sorted(id_set):
        p_df   = df_ref[df_ref["Id"] == pid]
        n_win  = len(p_df)
        n_pos  = p_df[TARGET_COL].sum()
        has_fb = int(n_pos > 0)
        rate   = f"{n_pos/n_win*100:.1f}%" if n_win > 0 else "—"
        print(f"  {split_label:<8} {str(pid):<15} {has_fb:<15} {n_win:<12} {n_pos:<12} {rate:>13}")
    n_win_total = len(split_df)
    n_pos_total = split_df[TARGET_COL].sum()
    rate_total  = f"{n_pos_total/n_win_total*100:.1f}%" if n_win_total > 0 else "—"
    print(f"  {'':8} {'TOTAL':<15} {'-':<15} {n_win_total:<12} {n_pos_total:<12} {rate_total:>13}")
    print(f"  {'':8} Total feedback events: {n_fb_events}")
    print()

# ─── Step 2: Data exploration — categorical value counts ──────────────────────
header("Categorical column distributions (1s dataset)")

for col in CAT_COLS:
    if col in df_ref.columns:
        vc = df_ref[col].value_counts(dropna=False)
        pct = (vc / len(df_ref) * 100).round(1)
        print(f"\n  {col}:")
        for val, count in vc.items():
            print(f"    {str(val):<35} {count:>6} rows  ({pct[val]}%)")

# ─── Step 3: Preprocessing function ──────────────────────────────────────────
def preprocess(df, train_ids, test_ids, fit_encoder=None):
    """
    Split, encode, and return X_train, X_test, y_train, y_test.
    fit_encoder: if provided, use this encoder (for test set consistency).
                 if None, fit a new one on train.
    Returns: X_train, X_test, y_train, y_test, encoder_info
    """
    train_df = df[df["Id"].isin(train_ids)].copy()
    test_df  = df[df["Id"].isin(test_ids)].copy()

    # Separate target
    y_train = train_df[TARGET_COL].astype(int)
    y_test  = test_df[TARGET_COL].astype(int)

    # Drop identifier and target columns
    drop = [c for c in DROP_COLS + [TARGET_COL] if c in train_df.columns]
    train_df = train_df.drop(columns=drop)
    test_df  = test_df.drop(columns=drop)

    # One-hot encode categorical columns
    cat_present = [c for c in CAT_COLS if c in train_df.columns]

    if fit_encoder is None:
        # Fit on train, apply to both
        train_df = pd.get_dummies(train_df, columns=cat_present, dummy_na=False)
        # Save column order from train
        train_cols = train_df.columns.tolist()
        test_df = pd.get_dummies(test_df, columns=cat_present, dummy_na=False)
        # Align test to train columns (fill missing with 0, drop extra)
        test_df = test_df.reindex(columns=train_cols, fill_value=0)
        encoder_info = train_cols
    else:
        train_cols = fit_encoder
        train_df = pd.get_dummies(train_df, columns=cat_present, dummy_na=False)
        train_df = train_df.reindex(columns=train_cols, fill_value=0)
        test_df  = pd.get_dummies(test_df,  columns=cat_present, dummy_na=False)
        test_df  = test_df.reindex(columns=train_cols, fill_value=0)
        encoder_info = train_cols

    return train_df, test_df, y_train, y_test, encoder_info

# ─── Step 4: Evaluation function ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name, window_size):
    """Compute and print all metrics at default and optimal threshold.
    Returns a result dict."""
    from sklearn.metrics import precision_recall_curve

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ── Find optimal threshold (maximizes F1) ─────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # thresholds has one fewer element than precisions/recalls
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0
    )
    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]

    # ── Evaluate at default threshold (0.5) ───────────────────────────────────
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    cm_default     = confusion_matrix(y_test, y_pred_default)

    # ── Evaluate at optimal threshold ─────────────────────────────────────────
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    cm_optimal     = confusion_matrix(y_test, y_pred_optimal)

    # ── Overall metrics (threshold-independent) ───────────────────────────────
    auc = roc_auc_score(y_test, y_pred_proba)
    ap  = average_precision_score(y_test, y_pred_proba)

    # ── Metrics at optimal threshold ──────────────────────────────────────────
    f1_opt   = f1_score(y_test, y_pred_optimal,   zero_division=0)
    prec_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
    rec_opt  = recall_score(y_test, y_pred_optimal,  zero_division=0)

    # ── Metrics at default threshold ──────────────────────────────────────────
    f1_def   = f1_score(y_test, y_pred_default,   zero_division=0)
    prec_def = precision_score(y_test, y_pred_default, zero_division=0)
    rec_def  = recall_score(y_test, y_pred_default,  zero_division=0)

    print(f"\n  {'─'*55}")
    print(f"  {model_name} | window={window_size}s")
    print(f"  {'─'*55}")
    print(f"  Threshold-independent:")
    print(f"    AUC-ROC:               {auc:.4f}")
    print(f"    Average Precision:     {ap:.4f}")
    print(f"  At default threshold (0.50):")
    print(f"    F1:        {f1_def:.4f}  |  Precision: {prec_def:.4f}  |  Recall: {rec_def:.4f}")
    print(f"    TP={cm_default[1,1]}  FP={cm_default[0,1]}  TN={cm_default[0,0]}  FN={cm_default[1,0]}")
    print(f"  At optimal threshold ({best_threshold:.3f}):")
    print(f"    F1:        {f1_opt:.4f}  |  Precision: {prec_opt:.4f}  |  Recall: {rec_opt:.4f}")
    print(f"    TP={cm_optimal[1,1]}  FP={cm_optimal[0,1]}  TN={cm_optimal[0,0]}  FN={cm_optimal[1,0]}")
    print(f"  Test set: {y_test.sum()} positive / {len(y_test)} total "
          f"({y_test.mean()*100:.1f}%)")

    # ── Probability distribution insight ──────────────────────────────────────
    pos_probs = y_pred_proba[y_test == 1]
    neg_probs = y_pred_proba[y_test == 0]
    print(f"  Predicted probabilities:")
    print(f"    Positive cases — mean: {pos_probs.mean():.3f}  "
          f"min: {pos_probs.min():.3f}  max: {pos_probs.max():.3f}")
    print(f"    Negative cases — mean: {neg_probs.mean():.3f}  "
          f"min: {neg_probs.min():.3f}  max: {neg_probs.max():.3f}")

    return {
        "model":          model_name,
        "window_sec":     window_size,
        "AUC_ROC":        round(auc,      4),
        "Avg_Prec":       round(ap,       4),
        # default threshold
        "F1_default":     round(f1_def,   4),
        "Prec_default":   round(prec_def, 4),
        "Rec_default":    round(rec_def,  4),
        # optimal threshold
        "best_threshold": round(best_threshold, 3),
        "F1_optimal":     round(f1_opt,   4),
        "Prec_optimal":   round(prec_opt, 4),
        "Rec_optimal":    round(rec_opt,  4),
        # confusion matrix at optimal threshold
        "TP": int(cm_optimal[1,1]),
        "FP": int(cm_optimal[0,1]),
        "TN": int(cm_optimal[0,0]),
        "FN": int(cm_optimal[1,0]),
    }

# ─── Step 5: Main loop — train and evaluate on each window size ───────────────
all_results = []

for w in WINDOW_SIZES:
    header(f"Window size: {w}s")

    df = pd.read_csv(os.path.join(DATA_DIR, f"windows_{w}s_clean.csv"))

    # Basic dataset info
    n_events      = df["event_id"].nunique()
    n_pos         = df[TARGET_COL].sum()
    n_total       = len(df)
    n_events_with_fb = (df.groupby("event_id")[TARGET_COL].sum() > 0).sum()

    print(f"  Rows:                    {n_total:,}")
    print(f"  Unique TL events:        {n_events:,}")
    print(f"  Events WITH feedback:    {n_events_with_fb:,} / {n_events:,}  "
          f"({n_events_with_fb/n_events*100:.1f}%)")
    print(f"  Events WITHOUT feedback: {n_events - n_events_with_fb:,} / {n_events:,}  "
          f"({(n_events - n_events_with_fb)/n_events*100:.1f}%)")
    print(f"  Positive windows (target=1): {n_pos:,} / {n_total:,}  "
          f"({n_pos/n_total*100:.2f}%)")
    print(f"  Negative windows (target=0): {n_total - n_pos:,} / {n_total:,}  "
          f"({(n_total-n_pos)/n_total*100:.2f}%)")
    print(f"  Imbalance ratio (neg/pos):   {(n_total-n_pos)/max(n_pos,1):.1f}:1")

    # Per-participant feedback summary (only for first window size)
    if w == WINDOW_SIZES[0]:
        print(f"\n  Per-participant breakdown (shown once for {w}s window only):")
        print(f"  {'Id':<15} {'n_events':<12} {'events_with_fb':<18} {'n_windows':<12} {'pos_windows':<13} {'pos_rate':>10}")
        print(f"  {'-'*15} {'-'*12} {'-'*18} {'-'*12} {'-'*13} {'-'*10}")
        for pid, p_df in df.groupby("Id"):
            p_events    = p_df["event_id"].nunique()
            p_fb_events = (p_df.groupby("event_id")[TARGET_COL].sum() > 0).sum()
            p_windows   = len(p_df)
            p_pos       = p_df[TARGET_COL].sum()
            p_rate      = f"{p_pos/p_windows*100:.1f}%" if p_windows > 0 else "—"
            print(f"  {str(pid):<15} {p_events:<12} {p_fb_events:<18} {p_windows:<12} {p_pos:<13} {p_rate:>10}")
        print()

    # Preprocess
    X_train, X_test, y_train, y_test, encoder_info = preprocess(
        df, train_ids, test_ids
    )

    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
    print(f"  Test positive rate:  {y_test.mean()*100:.1f}%")

    # Class imbalance weight
    n_neg = (y_train == 0).sum()
    n_pos_train = (y_train == 1).sum()
    scale_pos = round(n_neg / max(n_pos_train, 1), 2)
    print(f"  scale_pos_weight: {scale_pos}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight = scale_pos,
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE,
        verbosity        = 0,
        use_label_encoder= False,
    )
    xgb_model.fit(X_train, y_train)
    result_xgb = evaluate(xgb_model, X_test, y_test, "XGBoost", w)
    all_results.append(result_xgb)

    # Feature importance — top 10
    fi = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    print(f"\n  XGBoost top 10 features:")
    for feat, imp in fi.nlargest(10).items():
        print(f"    {feat:<45} {imp:.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        class_weight = "balanced",
        n_estimators = 100,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    rf_model.fit(X_train, y_train)
    result_rf = evaluate(rf_model, X_test, y_test, "RandomForest", w)
    all_results.append(result_rf)

    # Feature importance — top 10
    fi_rf = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    print(f"\n  Random Forest top 10 features:")
    for feat, imp in fi_rf.nlargest(10).items():
        print(f"    {feat:<45} {imp:.4f}")

# ─── Step 6: Comparison table ─────────────────────────────────────────────────
header("COMPARISON TABLE — All models × All window sizes")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(["window_sec", "model"]).reset_index(drop=True)

# Print formatted table
print("\n  ── Threshold-independent metrics ──")
col_order = ["model", "window_sec", "AUC_ROC", "Avg_Prec"]
print(results_df[col_order].to_string(index=False))

print("\n  ── At optimal threshold ──")
col_order2 = ["model", "window_sec", "best_threshold",
              "F1_optimal", "Prec_optimal", "Rec_optimal", "TP", "FP", "TN", "FN"]
print(results_df[col_order2].to_string(index=False))

print("\n  ── At default threshold (0.50) ──")
col_order3 = ["model", "window_sec", "F1_default", "Prec_default", "Rec_default"]
print(results_df[col_order3].to_string(index=False))

# Best model per metric
print("\n── Best combination per metric ──")
for metric in ["AUC_ROC", "Avg_Prec", "F1_optimal", "Prec_optimal", "Rec_optimal"]:
    best = results_df.loc[results_df[metric].idxmax()]
    print(f"  {metric:<18} → {best['model']} | window={best['window_sec']}s "
          f"| {metric}={best[metric]:.4f}")

print("\n✨ Done!")