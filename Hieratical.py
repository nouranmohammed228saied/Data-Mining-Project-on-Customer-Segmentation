import pandas as pd
import matplotlib.pyplot as plt#visualize
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as sch# find the roots
data = pd.read_csv("Mall_Customers.csv")
labelEncoder=LabelEncoder()
labelEncoder.fit(data['Gender'] ) #labelEncoder takes data and fit
data['Gender'] =labelEncoder.transform(data['Gender'] ) #then transform
x = data['Annual Income (k$)']
x=data.iloc[:,[3,4]].values #colums
#We create an instance of AgglomerativeClustering using the euclidean distance as the measure of distance between points and ward linkage to calculate the proximity of clusters.
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()
##########################################
Agglomerative = AgglomerativeClustering().fit(data.iloc[:,1:])
print (Agglomerative.labels_)
agg=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
Agg=agg.fit_predict(data.loc[:,["Age","Spending Score (1-100)"]].values)
#each cluster with two columns
plt.scatter(data.loc[:,["Age","Spending Score (1-100)"]].values[Agg==0, 0],data.loc[:,["Age","Spending Score (1-100)"]].values[Agg==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(data.loc[:,["Age","Spending Score (1-100)"]].values[Agg==1, 0],data.loc[:,["Age","Spending Score (1-100)"]].values[Agg==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(data.loc[:,["Age","Spending Score (1-100)"]].values[Agg==2, 0],data.loc[:,["Age","Spending Score (1-100)"]].values[Agg==2, 1], s=100, c='green', label ='Cluster 3')
plt.title("Hieratical")
plt.xlabel('age')
plt.ylabel('Spending Score (1-100)')
plt.show()