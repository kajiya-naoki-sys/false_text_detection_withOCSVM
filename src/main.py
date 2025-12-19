# src/main.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

import lightgbm as lgb
import joblib


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train LightGBM on text (TF-IDF -> LGBMClassifier).")
    ap.add_argument("--csv", type=str, default="outputs/synthetic_misinfo.csv",
                    help="学習用CSV（text,label列がある想定）")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # TF-IDF
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--max_features", type=int, default=200_000)

    # LightGBM
    ap.add_argument("--n_estimators", type=int, default=5000)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--min_child_samples", type=int, default=20)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)

    # outputs
    ap.add_argument("--out_dir", type=str, default="outputs",
                    help="モデル等の保存先ディレクトリ")
    ap.add_argument("--model_name", type=str, default="lgbm_text_model.joblib")
    ap.add_argument("--tfidf_name", type=str, default="tfidf_vectorizer.joblib")
    ap.add_argument("--labelenc_name", type=str, default="label_encoder.joblib")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: '{args.text_col}', '{args.label_col}'")

    X = df[args.text_col].astype(str).fillna("")
    y_raw = df[args.label_col]

    # label encoding (string labels are OK)
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    # split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features
    )
    Xtr = tfidf.fit_transform(X_train)
    Xva = tfidf.transform(X_valid)

    n_classes = len(le.classes_)
    objective = "multiclass" if n_classes > 2 else "binary"
    eval_metric = "multi_logloss" if n_classes > 2 else "binary_logloss"

    # LightGBM
    model = lgb.LGBMClassifier(
        objective=objective,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=-1
    )

    model.fit(
        Xtr,
        y_train,
        eval_set=[(Xva, y_valid)],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True)]
    )

    # evaluation
    y_pred = model.predict(Xva)

    print("=== classes ===")
    print(list(le.classes_))
    print("\n=== classification report ===")
    print(classification_report(y_valid, y_pred, target_names=le.classes_))
    print("=== confusion matrix ===")
    print(confusion_matrix(y_valid, y_pred))

    # save artifacts
    model_path = out_dir / args.model_name
    tfidf_path = out_dir / args.tfidf_name
    le_path = out_dir / args.labelenc_name

    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(le, le_path)

    print("\n=== saved ===")
    print(f"model: {model_path.resolve()}")
    print(f"tfidf: {tfidf_path.resolve()}")
    print(f"label_encoder: {le_path.resolve()}")

    # quick demo prediction
    demo_texts = [
        "BREAKING! supposedly, Northbridge is hiding the truth about water quality reporting. Share this now!",
        "[Dec 3, 2025] Civic Data Office in Silverhaven released a published dataset confirming weather advisories."
    ]
    demo_X = tfidf.transform(demo_texts)
    demo_pred = model.predict(demo_X)
    print("\n=== demo predictions ===")
    for t, p in zip(demo_texts, demo_pred):
        print(f"- {t}\n  -> {le.inverse_transform([p])[0]}")


if __name__ == "__main__":
    main()
