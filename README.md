# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: GOKUL S
RegisterNumber:  212224230075
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
```
## Output:

## DATA.HEAD()

<img width="664" height="134" alt="image" src="https://github.com/user-attachments/assets/13dcbd33-b4d2-4752-bb0d-00378f394f81" />

## DATA.INF0()

<img width="551" height="275" alt="image" src="https://github.com/user-attachments/assets/1aba1c6c-222c-48e0-8287-6cc90d988706" />

## DATA.ISNULL().SUM()

<img width="359" height="161" alt="image" src="https://github.com/user-attachments/assets/b85c8618-d261-43f0-912c-72b939e2b9d6" />

## PLOT USING ELBOW METHOD

<img width="757" height="602" alt="image" src="https://github.com/user-attachments/assets/fd81858f-162c-4381-8b17-2a77ddc3b02a" />

## CUSTOMER SEGMENT

<img width="792" height="602" alt="image" src="https://github.com/user-attachments/assets/03e12838-6203-4ca8-bf7e-166b7bbe786b" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
