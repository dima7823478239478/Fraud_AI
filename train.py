import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import joblib
# Кривая обучения для Random Forest
from sklearn.model_selection import validation_curve


# ========== 1. ЗАГРУЗКА ПОДГОТОВЛЕННЫХ ДАННЫХ ==========
print("="*50)
print("1. ЗАГРУЗКА ДАННЫХ")
print("="*50)

# Загружаем train/test данные
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Разделяем на признаки и целевую переменную
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

print(f"Train: {X_train.shape}, Fraud: {y_train.mean()*100:.3f}%")
print(f"Test: {X_test.shape}, Fraud: {y_test.mean()*100:.3f}%")

# ========== 2. ОБУЧЕНИЕ МОДЕЛИ ==========
print("\n" + "="*50)
print("2. ОБУЧЕНИЕ RANDOM FOREST")
print("="*50)

# Создаем модель с параметрами для дисбаланса
model = RandomForestClassifier(
    n_estimators=100,        # количество деревьев
    max_depth=10,            # максимальная глубина (чтобы не переобучаться)
    min_samples_split=10,    # минимальное число样本 для разделения
    min_samples_leaf=5,      # минимальное число样本 в листе
    class_weight='balanced', # важно! автоматически учитывает дисбаланс
    random_state=42,
    n_jobs=-1                # использовать все ядра процессора
)

# Обучаем
model.fit(X_train, y_train)
print("✅ Модель обучена!")

# ========== 3. ОЦЕНКА НА ТЕСТЕ ==========
print("\n" + "="*50)
print("3. ОЦЕНКА МОДЕЛИ")
print("="*50)

# Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # вероятность класса 1

# Метрики
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\n🎯 ROC-AUC: {roc_auc:.4f}")

# ========== 4. CONFUSION MATRIX ==========
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ========== 5. ROC-КРИВАЯ ==========
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ========== 6. ВАЖНОСТЬ ПРИЗНАКОВ ==========
print("\n" + "="*50)
print("4. ВАЖНОСТЬ ПРИЗНАКОВ")
print("="*50)

# Получаем важность
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-15 самых важных признаков:")
print(importance.head(15))

# Визуализация
plt.figure(figsize=(10, 8))
plt.barh(importance.head(15)['feature'], importance.head(15)['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ========== 7. КРИВАЯ ОБУЧЕНИЯ (VALIDATION CURVE) ==========
print("\n" + "="*50)
print("7. КРИВАЯ ОБУЧЕНИЯ")
print("="*50)

# Определяем диапазон количества деревьев для проверки
n_estimators_range = [10, 50, 100, 150, 200]

# Считаем validation curve
train_scores, val_scores = validation_curve(
    RandomForestClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    X_train, y_train,
    param_name="n_estimators",
    param_range=n_estimators_range,
    cv=3,  # 3-кратная кросс-валидация
    scoring='roc_auc',
    n_jobs=-1
)

print("Расчет завершен!")

# График кривой обучения
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores.mean(axis=1), 'b-', label='Train', linewidth=2)
plt.plot(n_estimators_range, val_scores.mean(axis=1), 'r-', label='Validation', linewidth=2)
plt.fill_between(n_estimators_range,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1),
                 alpha=0.2, color='r', label='Val ± std')
plt.fill_between(n_estimators_range,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1),
                 alpha=0.2, color='b', label='Train ± std')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('ROC-AUC')
plt.title('Validation Curve for Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ========== 8. АНАЛИЗ ОШИБОК ==========
print("\n" + "="*50)
print("8. АНАЛИЗ ОШИБОК")
print("="*50)

# Находим ошибки
test_data_with_pred = test_data.copy()
test_data_with_pred['predicted'] = y_pred
test_data_with_pred['probability'] = y_proba

# Ложноположительные (нормальные, которые модель назвала мошенническими)
false_positives = test_data_with_pred[(test_data_with_pred['Class']==0) &
                                      (test_data_with_pred['predicted']==1)]
print(f"Ложноположительных: {len(false_positives)}")

# Ложноотрицательные (мошеннические, которые модель пропустила)
false_negatives = test_data_with_pred[(test_data_with_pred['Class']==1) &
                                      (test_data_with_pred['predicted']==0)]
print(f"Ложноотрицательных: {len(false_negatives)}")

if len(false_negatives) > 0:
    print("\nПримеры пропущенных мошенничеств:")
    print(false_negatives[['V4', 'V11', 'V2', 'Amount_scaled', 'probability']].head())

# ========== 9. СОХРАНЕНИЕ МОДЕЛИ ==========
print("\n" + "="*50)
print("9. СОХРАНЕНИЕ МОДЕЛИ")
print("="*50)

joblib.dump(model, 'fraud_detection_model.pkl')
print("✅ Модель сохранена в 'fraud_detection_model.pkl'")

# Сохраняем также список важных признаков
importance.to_csv('feature_importance.csv', index=False)
print("✅ Важность признаков сохранена в 'feature_importance.csv'")