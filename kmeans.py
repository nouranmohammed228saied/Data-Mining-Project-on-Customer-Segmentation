import pandas as pd
import matplotlib.pyplot as plt#visualize
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
########################
data = pd.read_csv("Mall_Customers.csv")#call file
labelEncoder=LabelEncoder()#convert data to number
labelEncoder.fit(data['Gender'] )
data['Gender'] =labelEncoder.transform(data['Gender'] )
Kmeans=KMeans(n_clusters=3).fit(data.iloc[:,1:])#skip first column in data(ID) ?مش بيدخل في analysis data فمش محتاجينه في ال result
#no of cluster =3 ,  اقل حاجه 2 واكبر حاجه عدد ال obj اللي عندي يعني ميزيدش عن  4
print(Kmeans.labels_)
########################visualize(1) "EUCLIDIAN " minimize distance between points in cluster
n=data.loc[:,["Age","Spending Score (1-100)"]].values #specify rows and columns based on their row and column labels.
kmean=KMeans(n_clusters=3)
KMEANS=kmean.fit_predict(data.loc[:,["Age","Spending Score (1-100)"]].values)
#are used to observe relationship between variables and uses dots to represent the relationship between them
plt.scatter(n[:,0],n[:,1],c=kmean.labels_,cmap='rainbow')
#we take "0"age,"1" spending score
# cluster_centers to fit data
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],color='black')
plt.title("Kmeans")
plt.xlabel('age')
plt.ylabel('Spending Score (1-100)')
plt.show()
##############################visualize(2)
n_samples = 4000 #"densisty of points "
n_components = 3 #"num of centroids" at least 2, max #of attrinutes
#make_blobs() function can be used to generate blobs of points with a Gaussian distribution.
#cluster_std :Standard deviation is a number that describes how spread out the values are
X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0)
X = X[:, ::-1]
# Calculate seeds from kmeans++
#initial centroids according to 3 clusters
centers_init, indices = kmeans_plusplus(X, n_clusters=3, random_state=0)
# Plot init seeds along side sample data
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
#ytrue take data from make plot  , this loop give colors for data
for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker="*", s=4) #s -> volume of point , marker the shape of data "points"
plt.scatter(centers_init[:, 0], centers_init[:, 1], c="b", s=30)  #"b" color code for centroids , s "volume of centroid "
plt.title("K-Means++")
plt.xticks([])
plt.yticks([])
plt.show()
##############################################
"""import nltk.data
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
import nltk
nltk.download()
# Program to measure the similarity between
# two sentences using cosine similarity.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()
X = 'Age'
Y = 'Age'

# tokenization
X_list = word_tokenize(X)
Y_list = word_tokenize(Y)

# sw contains the list of stopwords
sw = stopwords.words('english')
l1 = [];
l2 = []
# remove stop words from the string
X_set = {w for w in X_list if not w in sw}
Y_set = {w for w in Y_list if not w in sw}

# form a set containing keywords of both strings
rvector = X_set.union(Y_set)
for w in rvector:
    if w in X_set:
        l1.append(1)  # create a vector
    else:
        l1.append(0)
    if w in Y_set:
        l2.append(1)
    else:
        l2.append(0)
c = 0
# cosine formula
for i in range(len(rvector)):
    c += l1[i] * l2[i]
cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
print("similarity: ", cosine)"""