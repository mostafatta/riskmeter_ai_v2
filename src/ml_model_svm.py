import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ==========================================
# Rolling Window Validation
# ==========================================
def rolling_window_cv(n_samples, n_splits=5, test_size=0.15):
    """
    Rolling Window (Walk-Forward) Cross-Validation.
    تحترم الترتيب الزمني: التدريب دائماً على الماضي والاختبار على المستقبل.
    """
    test_len    = int(n_samples * test_size)
    total_train = n_samples - (n_splits * test_len)

    if total_train <= 0:
        raise ValueError("n_splits كبير جداً مقارنة بحجم البيانات")

    folds = []
    for i in range(n_splits):
        train_end  = total_train + i * test_len
        test_start = train_end
        test_end   = test_start + test_len
        folds.append((list(range(0, train_end)), list(range(test_start, test_end))))

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

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features: {expected_features}")

    # ==========================================
    # 2. Encode Labels & Scale Features
    # SVM حساس جداً للـ scale لذلك StandardScaler ضروري
    # ==========================================
    le = LabelEncoder()
    y  = le.fit_transform(df['Risk_Category'].values)

    scaler  = StandardScaler()
    X       = scaler.fit_transform(df[expected_features].values)

    print(f"Target classes: {le.classes_.tolist()}\n")

    # ==========================================
    # 3. Rolling Window Cross-Validation
    # ==========================================
    print("=" * 50)
    print("  ROLLING WINDOW CROSS-VALIDATION  (SVM)")
    print("=" * 50)

    n_splits = 5
    folds    = rolling_window_cv(len(X), n_splits=n_splits, test_size=0.15)

    # Hyperparameters grid للبحث
    # C: عقوبة الأخطاء | kernel: نوع الفصل | gamma: للـ RBF kernel
    param_grid = [
        {'C': 0.1,  'kernel': 'rbf',    'gamma': 'scale'},
        {'C': 1.0,  'kernel': 'rbf',    'gamma': 'scale'},
        {'C': 10.0, 'kernel': 'rbf',    'gamma': 'scale'},
        {'C': 1.0,  'kernel': 'rbf',    'gamma': 'auto'},
        {'C': 10.0, 'kernel': 'rbf',    'gamma': 'auto'},
        {'C': 1.0,  'kernel': 'linear', 'gamma': 'scale'},
        {'C': 10.0, 'kernel': 'linear', 'gamma': 'scale'},
        {'C': 1.0,  'kernel': 'poly',   'gamma': 'scale'},
    ]

    best_params  = None
    best_avg_acc = -1.0

    for params in param_grid:
        fold_accuracies = []

        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                random_state=42,
                probability=True      # لتفعيل predict_proba لاحقاً إن احتجنا
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_accuracies.append(accuracy_score(y_test, preds))

        avg_acc = np.mean(fold_accuracies)
        print(f"  C={params['C']:5.1f} | kernel={params['kernel']:6s} | gamma={params['gamma']:5s} "
              f"→ Avg Accuracy: {avg_acc:.2%}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params  = params

    print(f"\n  Best Params: {best_params}")
    print(f"  Best Rolling CV Accuracy: {best_avg_acc:.2%}")

    # ==========================================
    # 4. Train Final Model on All Data Except Last 15%
    # ==========================================
    test_size_n   = int(len(X) * 0.15)
    X_train_final = X[: -test_size_n]
    y_train_final = y[: -test_size_n]
    X_test_final  = X[-test_size_n :]
    y_test_final  = y[-test_size_n :]

    best_model = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma=best_params['gamma'],
        random_state=42,
        probability=True
    )
    best_model.fit(X_train_final, y_train_final)

    # ==========================================
    # 5. Final Test Evaluation
    # ==========================================
    y_pred   = best_model.predict(X_test_final)
    test_acc = accuracy_score(y_test_final, y_pred)

    print(f"\n{'='*50}")
    print(f"  TEST SET PERFORMANCE (FINAL — Last 15%)")
    print(f"{'='*50}")
    print(f"  Accuracy: {test_acc:.2%}\n")
    print(classification_report(y_test_final, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_final, y_pred))

    # ==========================================
    # 6. Save Model + Scaler + LabelEncoder
    # ==========================================
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models'
    )
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, "svm_rolling_window.pkl"))
    joblib.dump(scaler,     os.path.join(models_dir, "svm_scaler.pkl"))
    joblib.dump(le,         os.path.join(models_dir, "svm_label_encoder.pkl"))

    print(f"\nModel saved to {models_dir}/svm_rolling_window.pkl")
    print(f"Scaler saved to {models_dir}/svm_scaler.pkl")
    print(f"LabelEncoder saved to {models_dir}/svm_label_encoder.pkl")

    return best_model


if __name__ == "__main__":
    train_model()
