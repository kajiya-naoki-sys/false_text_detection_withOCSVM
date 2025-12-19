# src/ocsvm_train.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import joblib


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train one-class text model using ONLY normal(label0) for training.")
    ap.add_argument("--csv", type=str, default="outputs/synthetic_misinfo.csv", help="CSV with text,label columns.")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")

    ap.add_argument("--normal_label", type=int, default=0, help="Label treated as NORMAL (inlier).")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # TF-IDF
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--max_features", type=int, default=200_000)

    # Model choice
    ap.add_argument("--mode", type=str, default="linear_sgd", choices=["linear_sgd", "rbf_svd"])
    ap.add_argument("--nu", type=float, default=0.05, help="Upper bound on fraction of outliers (approx).")

    # linear_sgd (SGDOneClassSVM)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--learning_rate", type=str, default="optimal",
                    choices=["constant", "optimal", "invscaling", "adaptive"])
    ap.add_argument("--eta0", type=float, default=0.01)
    ap.add_argument("--power_t", type=float, default=0.5)
    ap.add_argument("--average", action="store_true")

    # rbf_svd
    ap.add_argument("--svd_components", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.1)

    # outputs
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--model_name", type=str, default="ocsvm_model.joblib")
    ap.add_argument("--tfidf_name", type=str, default="tfidf_vectorizer.joblib")
    ap.add_argument("--svd_name", type=str, default="svd.joblib")

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

    X_text = df[args.text_col].astype(str).fillna("").to_numpy()
    y = df[args.label_col].astype(int).to_numpy()

    # 評価用に holdout を作る（学習は normal のみ）
    X_train_text, X_valid_text, y_train, y_valid = train_test_split(
        X_text,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # 学習に使うのは normal だけ
    normal_mask_tr = (y_train == args.normal_label)
    n_normals = int(normal_mask_tr.sum())
    if n_normals < 10:
        raise ValueError(f"Too few normal samples in train split: {n_normals}")

    X_train_normal_text = X_train_text[normal_mask_tr]

    # ---- TF-IDF は normal だけで fit する（重要）----
    tfidf = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    Xtr_norm = tfidf.fit_transform(X_train_normal_text)  # fit は normal のみ
    Xva_all = tfidf.transform(X_valid_text)              # valid 全体は transform のみ

    svd = None

    if args.mode == "linear_sgd":
        from sklearn.linear_model import SGDOneClassSVM

        model = SGDOneClassSVM(
            nu=args.nu,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.seed,
            learning_rate=args.learning_rate,
            eta0=args.eta0,
            power_t=args.power_t,
            average=args.average,
        )
        model.fit(Xtr_norm)  # 学習データは normal のみ

        scores = model.decision_function(Xva_all)
        preds_inlier = model.predict(Xva_all)

    else:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.svm import OneClassSVM

        # SVD も normal だけで fit する（重要）
        svd = TruncatedSVD(n_components=args.svd_components, random_state=args.seed)
        Xtr_norm_s = svd.fit_transform(Xtr_norm)
        Xva_all_s = svd.transform(Xva_all)

        model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)
        model.fit(Xtr_norm_s)

        scores = model.decision_function(Xva_all_s)
        preds_inlier = model.predict(Xva_all_s)

    # ---- 二値化（inlier->normal_label，outlier->1-normal_label）----
    pred_label = np.where(preds_inlier == 1, args.normal_label, 1 - args.normal_label)

    print("=== Setting ===")
    print(f"mode={args.mode}，normal_label={args.normal_label}，nu={args.nu}")
    print(f"train normals={n_normals}／train total={len(y_train)}")
    print("TF-IDF fit は normal のみである．")

    print("\n=== confusion matrix ===")
    print(confusion_matrix(y_valid, pred_label))

    print("\n=== classification report ===")
    print(classification_report(y_valid, pred_label, target_names=["label0", "label1"]))

    # AUC は評価用ラベルがある場合のみ意味がある
    try:
        auc = roc_auc_score(y_valid, -scores)  # anomaly_score = -inlier_score
        print("\n=== ROC-AUC（label!=normal を anomaly とみなす） ===")
        print(f"{auc:.6f}")
    except Exception as e:
        print("\nROC-AUC could not be computed:", e)

    # save
    model_path = out_dir / args.model_name
    tfidf_path = out_dir / args.tfidf_name
    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    if svd is not None:
        joblib.dump(svd, out_dir / args.svd_name)

    print("\n=== saved ===")
    print(f"model: {model_path.resolve()}")
    print(f"tfidf: {tfidf_path.resolve()}")
    if svd is not None:
        print(f"svd: {(out_dir / args.svd_name).resolve()}")


if __name__ == "__main__":
    main()
