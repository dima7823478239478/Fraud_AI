import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ========== 1. ЗАГРУЗКА ДАННЫХ ==========
row_data = pd.read_csv('/Users/dmitrii/PycharmProjects/fraud_AI/creditcard.csv')
print("="*50)
print("1. ПЕРВИЧНЫЙ АНАЛИЗ")
print("="*50)
print(f"Форма данных: {row_data.shape}")
print(f"Колонки: {row_data.columns.tolist()}")
print(f"Пропуски: {row_data.isnull().sum().sum()}")
print(f"\nРаспределение классов:\n{row_data['Class'].value_counts()}")
print(f"Процент мошенничеств: {row_data['Class'].mean()*100:.3f}%")

# ========== 2. АНАЛИЗ СУММ ==========
print("\n" + "="*50)
print("2. АНАЛИЗ СУММ")
print("="*50)
print(row_data['Amount'].describe())

normal = row_data[row_data['Class']==0]['Amount']
fraud = row_data[row_data['Class']==1]['Amount']
print(f"\nСредняя сумма нормальной: {normal.mean():.2f}")
print(f"Средняя сумма мошеннической: {fraud.mean():.2f}")

# ========== 3. АНАЛИЗ ВРЕМЕНИ ==========
print("\n" + "="*50)
print("3. АНАЛИЗ ВРЕМЕНИ")
print("="*50)
row_data['Hour'] = row_data['Time'] / 3600

normal_hours = row_data[row_data['Class']==0]['Hour']
fraud_hours = row_data[row_data['Class']==1]['Hour']

print("НОРМАЛЬНЫЕ ТРАНЗАКЦИИ:")
print(f"Средний час: {normal_hours.mean():.2f}")
print(f"Медиана: {normal_hours.median():.2f}")
print(f"Стандартное отклонение: {normal_hours.std():.2f}")

print("\nМОШЕННИЧЕСКИЕ ТРАНЗАКЦИИ:")
print(f"Средний час: {fraud_hours.mean():.2f}")
print(f"Медиана: {fraud_hours.median():.2f}")
print(f"Стандартное отклонение: {fraud_hours.std():.2f}")

# ========== 4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ==========
print("\n" + "="*50)
print("4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("="*50)
correlations = row_data.corr()['Class'].sort_values(ascending=False)
print("Топ-10 признаков по корреляции с Class:")
print(correlations.head(11))

# Сильно коррелирующие пары признаков
corr_matrix = row_data.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i],
                                     corr_matrix.columns[j],
                                     corr_matrix.iloc[i, j]))
print("\nСильно коррелирующие пары (|r| > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} — {pair[1]}: {pair[2]:.3f}")

# ========== 5. СРАВНЕНИЕ СРЕДНИХ ДЛЯ ТОП-ПРИЗНАКОВ ==========
print("\n" + "="*50)
print("5. СРАВНЕНИЕ СРЕДНИХ ДЛЯ ТОП-ПРИЗНАКОВ")
print("="*50)
top_features = correlations.head(6).index.tolist()[1:]  # первые 5 после Class
print("Средние значения признаков для разных классов:")
for col in top_features:
    normal_mean = row_data[row_data['Class']==0][col].mean()
    fraud_mean = row_data[row_data['Class']==1][col].mean()
    print(f"{col}: Normal={normal_mean:.4f}, Fraud={fraud_mean:.4f}, Разница={abs(normal_mean-fraud_mean):.4f}")

# ========== 6. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ ==========
print("\n" + "="*50)
print("6. ПОДГОТОВКА ДАННЫХ")
print("="*50)

X = row_data.drop('Class', axis=1)
y = row_data['Class']

# Масштабирование Amount
scaler = StandardScaler()
X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])

# Удаляем исходные колонки
X = X.drop(['Amount', 'Time', 'Hour'], axis=1, errors='ignore')

print(f"Количество признаков после обработки: {X.shape[1]}")
print(f"Признаки: {X.columns.tolist()}")

# ========== 7. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ ==========
print("\n" + "="*50)
print("7. СОХРАНЕНИЕ ДАННЫХ")
print("="*50)

# Добавляем целевую переменную обратно для сохранения
processed_data = X.copy()
processed_data['Class'] = y.values

# Сохраняем в CSV
processed_data.to_csv('creditcard_processed.csv', index=False)
print(f"Сохранено {processed_data.shape[0]} строк, {processed_data.shape[1]} колонок")
print("Файл: creditcard_processed.csv")

# ========== 8. РАЗДЕЛЕНИЕ НА TREIN/TEST ==========
print("\n" + "="*50)
print("8. РАЗДЕЛЕНИЕ НА TREIN/TEST")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Fraud в train: {y_train.mean()*100:.3f}%")
print(f"Test: {X_test.shape}, Fraud в test: {y_test.mean()*100:.3f}%")

# Сохраняем также разделенные данные (опционально)
train_data = X_train.copy()
train_data['Class'] = y_train.values
train_data.to_csv('train_data.csv', index=False)

test_data = X_test.copy()
test_data['Class'] = y_test.values
test_data.to_csv('test_data.csv', index=False)

print("\n✅ Все данные сохранены!")
print("- creditcard_processed.csv — все обработанные данные")
print("- train_data.csv — обучающая выборка (80%)")
print("- test_data.csv — тестовая выборка (20%)")