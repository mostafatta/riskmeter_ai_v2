import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ==========================================
# Rolling Window Validation
# ==========================================
def rolling_window_cv(X, y, n_splits=5, test_size=0.15):
    """
    Rolling Window (Walk-Forward) Cross-Validation.
    مناسبة للـ time-series لأنها تحترم الترتيب الزمني:
    - كل fold يدرّب على بيانات الماضي ويختبر على المستقبل
    - لا يوجد data leakage من المستقبل للماضي
    """
    n = len(X)
    test_len = int(n * test_size)
    total_train = n - (n_splits * test_len)

    if total_train <= 0:
        raise ValueError("n_splits كبير جداً مقارنة بحجم البيانات")

    folds = []
    for i in range(n_splits):
        train_end = total_train + i * test_len
        test_start = train_end
        test_end = test_start + test_len

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        folds.append((train_idx, test_idx))

    return folds


def train_model():
    # ==========================================
    # 1. Load Dataset
    # ==========================================
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'processed', 'portfolio_dataset.csv'
    )

    if not os.path.exists(data_path):
        print("Error: Dataset not found! Run data_generator.py first.")
        return None

    df = pd.read_csv(data_path)

    # ==========================================
    # 2. Validate Required Columns
    # ==========================================
    expected_features = [
        'Portfolio_Volatility',
        'Portfolio_Beta',
        'Sector_Volatility',
        'Sector_Beta',
        'Diversification_Index',
        'Market_Cap_Score',
    ]

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return None

    X = df[expected_features].values
    y = df['Risk_Category'].values

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features: {expected_features}")
    print(f"Target classes: {np.unique(y).tolist()}\n")

    # ==========================================
    # 3. Rolling Window Cross-Validation
    # بدلاً من GridSearchCV العادي الذي يخلط البيانات
    # نستخدم rolling window للحفاظ على الترتيب الزمني
    # ==========================================
    print("=" * 50)
    print("  ROLLING WINDOW CROSS-VALIDATION")
    print("=" * 50)

    n_splits = 5
    folds = rolling_window_cv(X, y, n_splits=n_splits, test_size=0.15)

    param_grid = [
        {'n_estimators': 100, 'max_depth': 5,    'min_samples_split': 2,  'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth': 10,   'min_samples_split': 5,  'min_samples_leaf': 2},
        {'n_estimators': 300, 'max_depth': 15,   'min_samples_split': 10, 'min_samples_leaf': 4},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,  'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth': 10,   'min_samples_split': 5,  'min_samples_leaf': 2},
    ]

    best_params = None
    best_avg_acc = -1.0

    for params in param_grid:
        fold_accuracies = []
        for fold_num, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_accuracies.append(accuracy_score(y_test, preds))

        avg_acc = np.mean(fold_accuracies)
        print(f"  Params: n_est={params['n_estimators']:3d} | depth={str(params['max_depth']):4s} "
              f"| split={params['min_samples_split']} | leaf={params['min_samples_leaf']} "
              f"→ Avg Accuracy: {avg_acc:.2%}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params = params

    print(f"\n  Best Params: {best_params}")
    print(f"  Best Rolling CV Accuracy: {best_avg_acc:.2%}")

    # ==========================================
    # 4. Train Final Model on All Data Except Last 15%
    # ==========================================
    test_size = int(len(X) * 0.15)
    X_train_final = X[: len(X) - test_size]
    y_train_final = y[: len(y) - test_size]
    X_test_final  = X[len(X) - test_size :]
    y_test_final  = y[len(y) - test_size :]

    best_model = RandomForestClassifier(random_state=42, **best_params)
    best_model.fit(X_train_final, y_train_final)

    # ==========================================
    # 5. Final Test Evaluation
    # ==========================================
    test_preds = best_model.predict(X_test_final)
    test_acc   = accuracy_score(y_test_final, test_preds)

    print(f"\n{'='*50}")
    print(f"  TEST SET PERFORMANCE (FINAL — Last 15%)")
    print(f"{'='*50}")
    print(f"  Accuracy: {test_acc:.2%}\n")
    print(classification_report(y_test_final, test_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_final, test_preds))

    # ==========================================
    # 6. Feature Importance
    # ==========================================
    print(f"\n{'='*50}")
    print("  FEATURE IMPORTANCE")
    print(f"{'='*50}")
    for feat, imp in sorted(
        zip(expected_features, best_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(imp * 50)
        print(f"  {feat:30s} {imp:.4f} {bar}")

    # ==========================================
    # 7. Save Model
    # ==========================================
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models'
    )
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rf_rolling_window.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")

    return best_model


if __name__ == "__main__":
    train_model()
