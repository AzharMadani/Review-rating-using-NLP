Regression and Classification and Clustering is very easy on Relational datasets. I have applied Classification and Clustering on Continuous, Categorical as well as Image datasets.

NLP coupled with ML Some major use cases of NLP applications created via ML are:
1. Translation systems
2. Chatbots
3. Sentiment Analysis etc

I am positive that clustering works on continuous & categorical data but does it work on textual data as well?
Ans: We'll find out (Yes)

Data preprocessing

Type of data = images ---> Image preprocessing

Type of data = text -----> Natural language processing

We need to learn the basic NLP techniques to cluster textual data

Prescription: Blood sugar problem, eat leafy vegetables, breatthing exercises etc etc ---> DIAB, ALL, TYPE2 etc

Multiclass multilabel classification

These are common words used to describe good food or positive review But there would be 1 guy out of 1000 reviews who would use a fancy word to showcase simple concepts

supercalifragilisticexpialidocious -> 1 column in Sparse Matrix

lot of fancy words like this which would unneccesarily increase the columns in our dataset

The max_features argument would keep only 1000 most frequent words in your dataset ---> DR technique or hack

Tradeoff --> less max features means speed but less accuracy more max features means accuracy but less speed
