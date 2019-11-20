# Recipe Data Analysis using Pyspark


In this task, I will collect recipe data and use the ‘modified’ file which cleaned list of ingredients as compared to the ‘original’ file. Before analyzing more deeply, I try to get familiar with the data by understanding it well. In Task 1, I fix a dictionary of ingredients. Then basically each ingredient will become one dimension and we can represent each ingredient array as a sparse vector (of 0s and 1s). Locations of ones will correspond to the ingredients in a given recipe and map each ingredient array into these sparse vectors. I use ‘CountVectorizer’ to convert the list of ingredients to sparse vectors. ‘CountVectorizer’ converts the list of tokens above to vectors of token counts. In Task 2 I cluster the given data and characterize each cluster. I observed that in 1 cluster you have majority of recipes from certain categories, high ratings, low calories or such. In this task, I found out how to represent the quality of the clustering. And, because the range of features are so large, clustering the original data set is so biased and the result is not good. So, it is needed to normalize the data somehow to have more meaningful clusters. After normalizing the data set, I can get the meaningful results. In Task 3, I process the data to come up with some classes and create corresponding labels. Because the given data is unlabeled, I made three different targets: Rating labels(0, 1, 2, 3, 4, 5, 6, 7), Protein labels(Low protein: 0, High protein: 1), Fat labels (Low fat: 0, High fat: 1). After finishing the data pre-process, I split the data randomly into training and testing part (80% training, 20% testing). And, I train different kinds of Machine learning models:Logistic Regression, Naive Bayes, Decision tree, Random Forest, and Linear Support Vector Machine. The testing accuracy of your model is not high in the first time because the target variables are imbalanced and features are needed to normalized. For Logistic regression and SVM, I can check the lrModel.coefficientMatrix from the lrModel and find the large values. The large values are corresponding to the importance features. Their meaning is that the larger value is, the more it can be important. The testing accuracy be improved further by using Decision tree models. Specifically, Random forest and Gradient-Boosted tree are better suited for certain types of input data. And, after normalizing the data set through Normalizer, StandardScaler, and MinMaxScaler, the result is improved further.

## 1 Introduction
We have recipe data set with unlabeled data. In this task, my objective is to get some meaningful insight from unstructured data set. I explore the data set and check the relationships between the features through statistical information and data visualizations. Using clustering each features, I identify the characteristic of each clustering and can suggest the food company or food-related people manage their resource more efficiently. Through this task, I try to answer the question: What story does it tell about the relationships between the variables and get some meaningful insights/discover the patterns from the recipe data set.

## 2 Data understanding (Data descriptive)
Our data set is related with recipe for 18,187 with 11 features. Original recipe data set has 20,130 rows and modified recipe data set has 18,187. And, there are 11 features: calories, categories, date, desc, directions, fat, ingredients, rating, sodium, title

## 3 Exploratory Data Analysis 3.1 Descriptive Statistical Analysis
3.1 Descriptive Statistical Analysis
Let’s first take a look at the variables by utilizing a description method. The describe function automatically computes basic statistics for all continuous variables. Any null values are automatically skipped in these statistics. This will show:
- the mean
- the standard deviation (std) - the minimum value
- the maximum value

3.2 Preparing Data for Machine Learning
Real data are rarely clean and complicated. Normally, they tend to be incomplete, noisy, and inconsistent and it is super important task to prepossess the data by handling missing values. Let’s identify missing values in the dataset.

## 4 Feature engineering
In most of tasks, I use Dataframe for analyzing the data set. And, after choosing features to analyze and handling the missing values, I change the text variables into a sparse vectors for using this meaningful data. We usually work with structured data but unstructured text data can have vital content for analysis. The next step is Feature Engineering. PySpark has made it so easy that we do not need to do much for extracting features.
1. Tokenize the ingredient
2. Apply CountVectorizer
3. StringIndexer
4. VectorAssembler
5. Pipeline

## 5 Clustering
This task is about task 2 and about K-means clustering. K-means is a type of unsupervised learning and one of the popular methods of clustering unlabelled data into k clusters.


Step 1: Assemble your features
Step 2: fit your KMeans model
Step 3: transform your initial dataframe to include cluster assignments
Step 4: Determine the optimal number of clusters for k-means clustering

## 6 Model Training and Evaluation
Since this is a classification problem with binary/multi-class response, the classification method includes Logistic regression, Naive bayes, Decision tree (Random forest, Gradient boosted tree), Support vector machines (grid search) and neural network algorithms. All my models are supervised machine learning algorithms. And, because the given data is unlabeled, I need to come up with some classes and create corresponding labels. Regarding creating labels in this task, I use the multiple rating, high-protein content vs low-protein content, high-fat content vs low-fat content as a target of my models. I used different targets in order to check all the cases and learn the techniques to improve the models. Therefore, I build the classification model based on each target and predict the outcome (e.g predict if the food contain high-protein content or low-protein content, predict the rating of foods based on the ingredients, sodium, fat, and so on.) Before building the models, we need to split data into 80% training, 20% testing. We build the models based on training data set and validate our models through test data set. This task is super important because it helps us avoid over-fitting and accurately measure how well our model generalizes to new data. In this task, I attempt to improve the testing accuracy by using any other algorithms. This is because it is possible that certain algorithms are better suited for certain types of input data. In this task, Decision tree models (including Random forest, Gradient Boosted tree) perform the best. (Over 94%) The reason is that Decision trees are able to handle both continuous/categorical variable and there are missing values in the data set. Decision trees are robust for this data set.
