import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Regression and Classification and Clustering is very easy on Relational
# datasets. And in the Case Study sessions, we have applied Classification
# and Clustering on Continuous, Categorical as well as Image datasets.

# NLP coupled with ML
# Some major use cases of NLP applications created via ML are:
# 1. Translation systems
# 2. Chatbots
# 3. Sentiment Analysis etc

# We are positive that clustering works on continuous & categorical
# data but does it work on textual data as well?
# Ans: We'll find out (Yes)

df = pd.read_csv('review.csv', sep = '\t')

# Data preprocessing
# Type of data = images ---> Image preprocessing
# Type of data = text -----> Natural language processing

# We need to learn the basic NLP techniques to cluster textual data

# Prescription: Blood sugar problem, eat leafy vegetables, montair lc
# tores, breatthing exercises etc etc ---> DIAB, ALL, TYPE2 etc
# Multiclass multilabel classification

# X = prescription
# y = [DIAB, ALL TYPE2] (multiple labels in the vector)
# y_pred = [DIAB, TYPE1, ALL]

# Applying NLP
# pip install nltk
# nltk.download('stopwords')

import nltk
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

clean_reviews = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    clean_reviews.append(review)

# Applying Bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 800)

# like
# love
# yummy
# delicious
# tasty
# mouth-watering
# scrumptious
# crunchy

# These are common words used to describe good food or positive 
# review
# But there would be 1 guy out of 1000 reviews who would use a fancy
# word to showcase simple concepts
# supercalifragilisticexpialidocious -> 1 column in Sparse Matrix
# lot of fancy words like this which would unneccesarily increase
# the columns in our dataset
# The max_features argument would keep only 1000 most frequent words
# in your dataset ---> DR technique or hack
# Tradeoff --> less max features means speed but less accuracy
# more max features means accuracy but less speed

X = cv.fit_transform(clean_reviews)
X = X.toarray()

# Clustering Analysis

from sklearn.cluster import KMeans

# Elbow Technique

wcv = []

for i in range(1, 11):
    km = KMeans(n_clusters = i)
    km.fit(X)
    wcv.append(km.inertia_)

plt.plot(range(1, 11), wcv)
plt.xlabel('K (No of clusters)')
plt.ylabel('WCV')
plt.title('Clustering Analysis')
plt.show()

# Silhouette Analysis

from sklearn.metrics import silhouette_score

silhouette = []

for i in range(2, 11):
    km = KMeans(n_clusters = i)
    km.fit(X)
    silhouette.append(silhouette_score(X, km.labels_))

plt.plot(range(2, 11), silhouette)
plt.xlabel('K (No of clusters)')
plt.ylabel('Silhouette Score')
plt.title('Clustering Analysis')
plt.show()

from yellowbrick.cluster import silhouette_visualizer
silhouette_visualizer(KMeans(3, random_state = 42), X, colors = 'yellowbrick')

# Insights: Both the clusters have a silhouette score higher than
# the average silhouette score.
# Clusters are imbalanced.

df['Liked'].value_counts()
# Target is balanced.
# Not the optimal case of clustering.

km_best = KMeans(n_clusters = 2)
y_label = km_best.fit_predict(X)

df['Cluster'] = y_label

# Inspect the cluster rules and experiment with other clustering
# algorithms as well.

# Divide and Conquer
# K-Means and HCA

# Neighborhood Analysis
# DBSCAN, Mean-Shift

# Probabilistic Analysis
# GMM (Gaussian Mixture Model)

from sklearn.datasets import make_moons
x, y = make_moons(n_samples = 300, noise = 0.05)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

km_test = KMeans(n_clusters = 2)
y_label_km = km_test.fit_predict(x)

plt.scatter(x[:, 0][y_label_km == 0], x[:, 1][y_label_km == 0])
plt.scatter(x[:, 0][y_label_km == 1], x[:, 1][y_label_km == 1])
plt.show()

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.15, min_samples = 5)
y_label_db = dbscan.fit_predict(x)
pd.Series(y_label_db).value_counts()

# plt.scatter(x[:, 0][y_label_db == -1], x[:, 1][y_label_db == -1])
plt.scatter(x[:, 0][y_label_db == 0], x[:, 1][y_label_db == 0])
plt.scatter(x[:, 0][y_label_db == 1], x[:, 1][y_label_db == 1])
plt.show()

# DBSCAN is better for high dimensional and irregular shapes data
# So let us apply dbscan on the cleaned reviews as well

dbscan = DBSCAN(eps = 2, min_samples = 7)
y_label_db = dbscan.fit_predict(X)
pd.Series(y_label_db).value_counts()

df['cluster_db'] = y_label_db

























