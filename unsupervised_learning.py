# Неконтролируемый: находит структуры в данных без каких-либо указаний.
#customer segmentation           - - - 07.10.24. - - -
# Unsupervised Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Create a simple dataset
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,2,27,28,29,30],
    'Annual_Income': [15,16,17,18,19,20,21,22,23,24,20,21,22,23,24,20,21,22,23,24,20,21,22,23,24,20,21,22,23,24],
    'Spending_Score': [39,81,6,77,40,76,6,94,3,72,39,81,6,77,40,76,6,94,3,72,39,81,6,77,40,76,6,94,3,72,]
}

# df - это сокращение от DataFrame в библиотеке pandas.

df = pd.DataFrame(data)   # converts this dictionary into a pandas DataFrame - преобразует этот словарь в фрейм данных pandas

#Features: Annual Income and Spending Score
X = df[['Annual_Income', 'Spending_Score']] # Selecting Features for Clustering - Выбор объектов для кластеризации

#Apply K-means with 3 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
# fit(X) — это шаг, на котором модель вычисляет центры кластеров для заданных данных.

#Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_  # Присвоение кластерных меток каждому клиенту

#Visualize the clusters
plt.figure(figsize=(8,6))  # Создает новую фигуру для графика, задавая размер графика

plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis') # viridis - Определяет цветовую схему

plt.title('Customer Clusters based on Annual Income and Spending Score')
plt.xlabel('Annual Income')   # Годовой доход x
plt.ylabel('Spending Score')  #Оценка расходов  y
plt.show()

#  c=df['Cluster'] - Окрашивает каждую точку в соответствии с назначенным ей кластером.