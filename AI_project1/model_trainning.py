import tempfile
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

tempfile.tempdir = os.path.join("D:", "temp")
os.makedirs(tempfile.tempdir, exist_ok=True)

def train_models(X, y, output_dir="models"):
    """训练模型并优化性能"""
    os.makedirs(output_dir, exist_ok=True)

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 特征选择（基于随机森林的重要性）
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold='mean'  # 选择重要性高于均值的特征
    )
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    joblib.dump(selector, os.path.join(output_dir, "feature_selector.pkl"))
    print(f"特征选择后维度: {X_train_selected.shape[1]} (原始: {X_train_scaled.shape[1]})")

    # 优化模型参数
    models = {
        "RandomForest": GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid={
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            cv=3, n_jobs=1
        ),
        "DecisionTree": GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid={
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            cv=3, n_jobs=-1
        ),
        "LogisticRegression": GridSearchCV(
            LogisticRegression(max_iter=500, random_state=42),
            param_grid={
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            cv=3, n_jobs=-1
        ),
        "NaiveBayes": GaussianNB()  # 朴素贝叶斯无参数可优化
    }

    best_model = None
    best_accuracy = 0
    model_results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")

        model.fit(X_train_selected, y_train)

        # 对于网格搜索，使用最佳估计器
        if hasattr(model, 'best_estimator_'):
            print(f"最佳参数: {model.best_params_}")
            y_pred = model.best_estimator_.predict(X_val_selected)
            current_model = model.best_estimator_
        else:
            y_pred = model.predict(X_val_selected)
            current_model = model

        accuracy = accuracy_score(y_val, y_pred)
        model_results[name] = accuracy

        print(f"准确率: {accuracy:.4f}")
        print(classification_report(y_val, y_pred, target_names=[
            "Black-grass", "Common wheat", "Loose Silky-bent",
            "Scentless Mayweed", "Sugar beet"
        ]))

        joblib.dump(current_model, os.path.join(output_dir, f"{name}_model.pkl"))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = current_model

    # 保存最佳模型和预处理工具
    joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    y_pred_best = best_model.predict(X_val_selected)
    cm = confusion_matrix(y_val, y_pred_best)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {max(model_results, key=model_results.get)}")
    plt.colorbar()

    classes = ["Black-grass", "Common wheat", "Loose Silky-bent",
               "Scentless Mayweed", "Sugar beet"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # 绘制模型比较图
    plt.figure(figsize=(10, 6))
    plt.bar(model_results.keys(), model_results.values(), color='skyblue')
    plt.title('Model Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, (name, acc) in enumerate(model_results.items()):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()

    print(f"\n最佳模型: {max(model_results, key=model_results.get)} (准确率: {best_accuracy:.4f})")

    return best_model, scaler, selector