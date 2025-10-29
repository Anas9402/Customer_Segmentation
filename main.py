import pandas as pd
import datetime
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df=pd.read_csv('customer_segmentation.csv').dropna()
df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'],dayfirst=True) #string to datetime convert using to_datatime.dayfirst 2025-08-01
df['Age']=2025-df['Year_Birth']
df['Total_Children']=df['Kidhome']+df['Teenhome']
spend_cols=['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']
df['Total_Spending']=df[spend_cols].sum(axis=1)
df['Customer_Since']=(pd.Timestamp('today')-df['Dt_Customer']).dt.days # type: ignore
X=df[['Age','Income','Total_Spending','NumWebPurchases','NumStorePurchases','NumWebVisitsMonth','Recency']].copy()

scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
kmeans=KMeans(n_clusters=6,random_state=42)
df['Cluster']=kmeans.fit_predict(x_scaled)
joblib.dump(kmeans,'model_kmeans.pkl')
joblib.dump(scaler,'scaler.pkl')

"""
wss=[]

for i in range(2,10):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(x_scaled)
    wss.append(kmeans.inertia_) 

plt.plot(range(2,10),wss,marker='o')
plt.title('Elbow method for find optimal k')
plt.xlabel('number of cluster')
plt.ylabel('wss inertia')
plt.show()
plt.tight_layout()
"""




