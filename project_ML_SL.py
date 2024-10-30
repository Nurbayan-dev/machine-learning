import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Years Abroad': [1, 3, 3, 4, 6, 8, 5, 2, 5, 7, 9, 10],
    'Expected Salary': [30, 45, 50, 60, 70, 80, 75, 40, 65, 85, 90, 100]  # Предполагаемая зарплата в тысячах долларов
}

# Преобразуем данные в массивы numpy
X = np.array(data['Years Abroad']).reshape(-1, 1)  # Признак: количество лет обучения за границей
y = np.array(data['Expected Salary'])               # Целевая переменная: ожидаемая зарплата

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(x_train, y_train)
# Делаем предсказания на тестовых данных
y_pred = model.predict([[7]])
print(y_pred)                      # бул жерди ручной коштук 7 жыл окуп келсе [78.39285714] табат дегенди чыгарып берди


# Вычисляем и выводим среднюю квадратичную ошибку (MSE) для оценки производительности модели
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# Создаем более широкий диапазон для линии предсказания
X_range_extended = np.linspace(X.min() - 2, X.max() + 2, 100).reshape(-1, 1)
y_range_pred = model.predict(X_range_extended)

# Визуализация фактических данных и предсказанных результатов
plt.figure(figsize=(8, 6))

# Фактические данные
plt.scatter(X, y, color='blue', label='Actual Data', s=100, alpha=0.6, edgecolor='black')

# Линия предсказаний
plt.plot(X_range_extended, y_range_pred, color='red', label='Predicted Line', linewidth=2)

# Добавляем заголовок и метки осей
plt.title('Relationship Between Years Abroad and Expected Salary', fontsize=16)
plt.xlabel('Years Studied Abroad', fontsize=12)
plt.ylabel('Expected Salary (in thousands USD)', fontsize=12)

# Включаем сетку для улучшения видимости данных
plt.grid(True)

# Добавляем легенду
plt.legend()

# Показать график
plt.show()
