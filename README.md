# Approaching-almost-any-Machine-Learning-Problem-Book
My notes of Approaching (Almost) any Machine Learning Problem book.

As the name of the book suggests, this book is about how you can approach any machine learning problem. It gives you a road map to any machine learning project, from setting up your Virtual Environment to Deploying your model.

## Table of Content

- [Chapter 1 - Setting up Your Working Environment](#chapter1)
- [Chapter 2 - Supervised vs Unsupervised Learning](#chapter2)
- [Chapter 3 - Cross Validation](#chapter3)
- [Chapter 4 - Evaluation Metrics](#chapter4)


<a name = "chapter1">
<h1>Chapter 1 - Setting up Your Working Environment</h1>
</a>

This chapter is about setting up working environment for your ML project. The first step to any ML project is creating a working environment for your project.

And the first step in setting up a working environment for an ML project is, installing python. 

### How to install python?
Just go to [Download Python](https://www.python.org/downloads/) page, and download a stable version of python and install it in your system.

Thats it!, now you have Python in your system.
You can open command prompt and type, "python or py" and hit enter, to check if it has installed or not. If it is installed you will see something like this.

![Alt text](https://i.stack.imgur.com/K5kSC.png "check python")


Now that we have installed python in our system, its time to set up a Virtual Environment.

### Why set up a new Virtual Environment?

- To keep Python's Built-in packages separate from third party packages.
- To install all our project dependancies (packages and libraries) in our working environment.
- Installing third party packages in Python environment can cause issues.
- Also installing dependancies in a virtual environment makes it easy to migrate our project to different system.

### How to set up a Virtual Environment?
[How to create a python Virtual Environments](https://docs.python.org/3/library/venv.html) this page provides you a very easy explanation for creating Python virtual environment. 

### Installing python packages

Once you have created a virtual environment, Activate the environment by running "activate.bat" from your environment directory. After activating your environment your environment name will show in the brackets, like this "(project_env)" in the picture below:

![Alt text](https://miro.medium.com/max/1095/1*oTcSPKxWdQe_jNh7yYDsNg.png "activate venv")


Once you have successfully activated your environment, just type "pip install package_name" and hit enter, the package will install. After installing all important packages like numpy, pandas, matplotlib, seaborn, scikit-learn, and jupyter lab. You are done with setting up your working environment. Now, we can begin writing programs for our project.

### Where to write Code?

Just open a fresh command prompt, activate your virtual environment and type "jupyter notebook" and hit enter. This will open Jupyter Notebook in your browser, here you can write your python code and also import all the packages you installed for your project.


<a name = "chapter2">
<h1>Chapter 2 - Supervised vs Unsupervised Learning</h1>
</a>

In this chapter we will learn about types of Machine Learning Problems. There are generally two types of Machine learning problems:
- Supervised Learning 
- Unsupervised Learning

And there are generally two types of data:
- **Supervised Data**: a data that always has one or multiple target variables.
- **Unsupervised Data**: a data that does not have any target variable.

So, a **Supervised Learning problem** is one where we are provided with *Supervised type of data* and are required to predict target/label for unknown samples by training a model on known samples.

and similarly, an **Unsupervised Learning problem** is one where we are provided with *Unsupervised type of data* and are required to predict the target/labels for whole data using unsupervised algorithms.

### Two types of Supervised Learning problems
- **Classification Problem**: When the data has categorical target variable(s) (when target variable has finite or discrete set of values), and we are required to predict the class of unknown sample.

- **Regression Problem**: When the data has target variable consisting of real values (or real numbers), and we are required to predict the target value for the unknown sample.

### Classification Problem Example
We are given a data consisting of images of cats and dogs, and the target variable is binary type (two categories cat or dog). We are then given unknown images (one without labels), our task is to classify the image if it is a cat image or dog image.

### Regression Problem Example
We are given a data of historical house prices with features like, presence of hospital, school or supermarket, distance to nearest public transport etc. The target variable of data is real valued (house prices), our task is to predict the price of a house with the help of given set of features. 

### How to approach an Unsupervised Learning problem
- **Clustering**: There are many clustering algorithms like, K-Means, DB-Scan, ART etc., which are used to deal with unsupervised data. As the name suggests, clustering algorithms takes data as input and form clusters of data which can then be labelled.

- **Decomposition Method**: Decomposition methods like, Principal Component Analysis (PCA) and t-Distributed Stochastic Neighborhood Embedding (t-SNE) are used for visualizing the data in lower dimensions. For Example: with the help of PCA or t-SNE we can decompose a data with hundereds of features to 2d data and easily visualize it. Check out this blog for better undersanding of the concept, [t-SNE Explanation with MNIST dataset](https://colah.github.io/posts/2014-10-Visualizing-MNIST/).


<a name = "chapter3">
<h1> Chapter 3 - Cross Validation</h1>
</a>

Before we do anything, we should first check which Cross Validation method is best for our dataset, and that is done by a little bit of data exploration.

but before that, What is Cross Validation?

### What is Cross Validation?
 Cross Validation is a step in model building process that ensures that our model fits the data accurately and also ensures that we do not overfit. We will come back to overfit later.

 Cross Validation method depends on the dataset:

 ### Types of Cross Validation
 - hold-out based validation
 - k-fold cross validation
 - stratified k-fold cross validation
 - leave-one-out cross validation
 - group k-fold cross validation

#### hold-out based validation
In this cross validation method we divide the whole data in two parts, and train model with one part of the data and validate/test with the other part.

- this cross validation method is opted when we have large amount of data.
- the samples of each class should be almost-eqaully distributed in data, i.e, there should big class imbalance in data.

#### k-fold cross validation
In this cross validation method we create "k folds" of data, and each fold has almost-equal distribution of data, then we treat each fold as validation data and use the rest to train model (in one-vs-rest fashion) one-by-one.

- this method is opted when we have small amount of data.
- and the data should not have big class imbalance.

#### stratified k-fold cross validation
It is similar to k-fold cross validation method but it also ensures that each fold must contain the samples from the class which has less number of samples.

- this method is opted when we have small amount of data.
- and the data have big class imbalance (one class dominates the other).

### Why cross validation first?
>Cross-validation is the first and most essential step when it comes to building machine learning models. If you want to do feature engineering, split your data first. If you're going to build models, split your data first. If you have a good cross validation scheme in which validation data is representative of training and real world data, you will be able to build a good machine learning model which is highly generalizable.


<a name = chapter4>
<h1>Chapter 4 - Evaluation Metrics</h1>
</a>

The next step in model building process is Evaluation. In this step we use the predictions of validation data to evaluate our model's performance. And for that we apply different type of evaluation metrics. What metric to use when depends on the dataset.

### Types of Evaluation Metrics
For classification problem we use following evaluation metrics:
- Accuracy Score
- Precision (P)
- Recall (R)
- F1 Score (F1)
- Area Under the ROC (Receiver Operating Characteristic) curve or simply Area Under the Curve (AUC)
- Log loss
- Precision at k (P@k)
- Average Precision at k (AP@k)
- Mean Average Precision at k (MAP@k)

For regression problem we use following evaluation metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Root Mean Squared Logarithmic Error (RMSLE)
- Mean Percentage Error (MPE)
- Mean Absolute Percentage Error (MAPE)
- R^2 (R-Squared )

Now, we will discuss each of these metrics in little detail.

### Accuracy Score
Accuracy is the simplest of evaluation metrics.
> Accuracy = *No. of correctly Classified points / Total number of data points* 

**Example:**
Consider a dataset with 200 images, 100 Images of X-Ray with disease (+ve), 100 Images of X-Ray witout disease (-ve)

Divide the dataset into two equal sets of 100 images

Training Set = (50 (+ve), 50 (-ve))

Validation Set = (50 (+ve), 50 (-ve))

*NOTE: When we have equal number of +ve class and -ve class datapoints we can use Accuracy Score, Precision, Recall and F1 Score as evaluation metric.*

Since, we have a dataset with equal distribution of negative and positive class, we can use Accuracy score as evaluation metric.

Now, suppose our model predicts the class of 90 images correctly as (-ve) class.

*ACCURACY = 90(-ve class) / 100(-ve class) = 90%*

We can say that our model is 90% accurate.

#### Where Accuracy Score fails?
Our Accuracy score metric fails when there is a significant imbalance in classes. To understand that let us modify the previous example a little bit.

Consider a dataset with 200 images, 180 Images of X-Ray with disease (+ve), 20 Images of X-Ray witout disease (-ve)

Divide the dataset into two equal sets of 100 images

Training Set = (90 (+ve), 10 (-ve))

Validation Set = (90 (+ve), 10 (-ve))

*NOTE: Notice that we have imbalance in classes, i.e, +ve class have more data points (180) than -ve class (20).*

Now, if you say, all images in our validation set are images of X-Ray with disease (+ve), what would be the Accuracy of this model?

You correctly classified 90% of the images so your model's Accuracy score is 90% (But that is useless!!)

*Our data was highly sqewed that is, the number of samples of one class outnumber the number of samples of the other class*

**Conlusion:** When data is highly skewed i.e, there is great imbalance in the classes then it is adviced not to use Accuracy Score as evaluation metric.

#### Accuracy Score implementation in python
```pyhton
    def accuracy(y_pred, y_true):
        count=0
        for yp, yt in zip(y_pred, y_true):
            if yp == yt:
                count+=1

        return count/len(y_pred)

```

### Precision (P), Recall (R), and F1 Score (F1)
We use precision, recall and/or f1 score where Accuracy metric fails, i.e, when there is huge imbalance in classes in the data. For understanding precision, recall and f1 score we need to know a few terms.

- **True Positive (TP)**: When the actual class of sample is *positive* and our model predict it as *positive*.
- **True Negative (TN)**: When the actual class of sample is *negative* and our model predict it as *negative*.
- **False Positive (FP)**: When the actual class of sample is *negative* and our model predict it as *positive*.
- **False Negative (FN)**: When the actual class of sample is *positive* and our model predict it as *negative*.

### Python Implementation for Precision
Firstly, we define the functions to calculate the TP, TN, FP, and FN.
```python
    def true_positive(y_true, y_pred):
        """Function to calculate True Positive
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: True Positive = number of correctly classified +ve labels

        """
        count = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                count+=1
        return count

    def true_negative(y_true, y_pred):
        """Function to calculate True Negative
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: True Negative = number of correctly classified -ve labels

        """
        count = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                count+=1
        return count

    def false_positive(y_true, y_pred):
        """Function to calculate False Positive
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: False Positive = number of wrongly classified +ve labels

        """
        count = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == 1 and yt == 0:
                count+=1
        return count

    def false_negative(y_true, y_pred):
        """Function to calculate False Negative
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: False Negative = number of wrongly classified -ve labels

        """
        count = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == 0 and yt == 1:
                count+=1
        return count
```
Now that we have defined functions for TP, TN, FP, and FN. We can Implement Precision.

Formula to Calculate Precision (P)
>*Precision (P) =  TP / (TP + FP)*

```python
    #Python code for Precision
    def precision(y_true, y_pred):
        """
        Function to calculate Precision of model
        :param y_true: list of true values
        :param y_pred: list of predicted values
        
        :return: Precision = TP /(TP + FP)
        
        """
        
        return true_positive(y_true, y_pred)/(true_positive(y_true, y_pred)\
                                            + false_positive(y_true, y_pred))

```

### Python Implementation for Recall
Recall is also known as Sensitivity or True Positive Rate (TPR).

Formula to Calculate Recall (R)
>*Recall (R) =  TP / (TP + FN)*

```python
    #Python code for calculating Recall
    def recall(y_true, y_pred):
        """
        Function to calculate Recall of model
        :param y_true: list of true values
        :param y_pred: list of predicted values
            
        :return: Recall = TP /(TP + FN)
            
        """
        
        # here we are using the previously defined functions for
        # true_positive and false_negative
        return true_positive(y_true, y_pred)/(true_positive(y_true, y_pred)\
                                            + false_negative(y_true, y_pred))
```

### Python Implementation for F1 Score
F1 Score is a combined representation of both Precision and Recall.

Formula to Calculate F1 Score (F1)
>*F1 Score (F1) =  2*P*R / (P + R)*

where, P: Precision; R: Recall

```python
    #Python code for calculating
    def f1_score(y_true, y_pred):
        """Function to calculate f1 Score
        :param y_true: a list of true values
        :param y_pred: a list of predicted values
        :return F1 Score: F1 Score = 2PR/(P+R)
        """
        
        # Here we are using previously defined precision and
        # recall functions
        return 2*precision(y_true, y_pred)*recall(y_true, y_pred)\
                /(precision(y_true, y_pred) + recall(y_true, y_pred))
```
