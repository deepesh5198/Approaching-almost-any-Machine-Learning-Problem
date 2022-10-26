# Approaching-almost-any-Machine-Learning-Problem-Book
My notes of Approaching (Almost) any Machine Learning Problem book.

As the name of the book suggests, this book is about how you can approach any machine learning problem. It gives you a road map to any machine learning project, from setting up your Virtual Environment to Deploying your model.

## Table of Content

- [Chapter 1 - Setting up Your Working Environment](#chapter1)
- [Chapter 2 - Supervised vs Unsupervised Learning](#chapter2)
- [Chapter 3 - Cross Validation](#chapter3)
- [Chapter 4 - Evaluation Metrics](#chapter4)
- [Chapter 5 - Arranging Machine Learning Projects](#chapter5)
- [Chapter 6 - Approaching Categorical Variables](#chapter6)


<a name = "chapter1">
<h1>Chapter 1 - Setting up Your Working Environment</h1>
</a>

This chapter is about setting up working environment for your ML project. The first step to any ML project is creating a working environment for your project.

And the first step in setting up a working environment for an ML project is, installing python. 

### How to install python?
Just go to [Download Python](https://www.python.org/downloads/) page, and download a stable version of python and install it in your system.

Thats it!, now you have Python in your system.
You can open command prompt and type, "python or py" and hit enter, to check if it has installed or not. If it is installed you will see something like this.

![Alt text](./images/check_python.png?web=raw "check python")


Now that we have installed python in our system, its time to set up a Virtual Environment.

### Why set up a new Virtual Environment?

- To keep Python's Built-in packages separate from third party packages.
- To install all our project dependancies (packages and libraries) in our working environment.
- Installing third party packages in Python environment can cause issues.
- Also installing dependancies in a virtual environment makes it easy to migrate our project to different system.

### How to set up a Virtual Environment?
[How to create a python Virtual Environments](https://docs.python.org/3/library/venv.html) this page provides you a very easy explanation for creating Python virtual environment. 

The image below shows how its done:
![Alt text](./images/creat_venv.png?web=raw "create_venv")

### Installing python packages

Once you have created a virtual environment, Activate the environment by running "activate.bat" from your environment directory. After activating your environment your environment name will show in the brackets, like this "(newenv)" in the picture below:

![Alt text](./images/activate_venv.png?web=raw "activate venv")


Once you have successfully activated your environment, just type "pip install package_name" and hit enter, the package will install. 

Like this:
![Alt text](./images/install_packages.png?web=raw "install packages")

After installing all important packages like numpy, pandas, matplotlib, seaborn, scikit-learn, and jupyter lab. You are done with setting up your working environment. Now, we can begin writing programs for our project.

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

*Our data was highly skewed that is, the number of samples of one class outnumber the number of samples of the other class*

**Conlusion:** When data is highly skewed i.e, there is great imbalance in the classes then it is adviced not to use Accuracy Score as evaluation metric.

#### Accuracy Score implementation in python
```python
    # Accuracy Score implementation in python
    def accuracy(y_pred, y_true):
        count=0
        for yp, yt in zip(y_pred, y_true):
            if yp == yt:
                count+=1

        return count/len(y_pred)
```
#### Scikit-Learn implementation of Accuracy Score
```python
    # Scikit-Learn implementation of Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true, y_pred)
```

### Precision (P), Recall (R), and F1 Score (F1)
We use precision, recall and/or f1 score where Accuracy metric fails, i.e, when there is huge imbalance in classes in the data. For understanding precision, recall and f1 score we need to know a few terms.

- **True Positive (TP)**: When the actual class of sample is *positive* and our model predict it as *positive*.
- **True Negative (TN)**: When the actual class of sample is *negative* and our model predict it as *negative*.
- **False Positive (FP)**: When the actual class of sample is *negative* and our model predict it as *positive*.
- **False Negative (FN)**: When the actual class of sample is *positive* and our model predict it as *negative*.

#### Python Implementation for Precision
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
#### Scikit-Learn implementation of Precision
```python
    # Scikit-Learn implementation of Precision
    from sklearn.metrics import precision_score
    precision_score(y_true, y_pred)
```

#### Python Implementation for Recall
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
#### Scikit-Learn implementation of Recall
```python
    # Scikit-Learn implementation of Recall
    from sklearn.metrics import recall_score
    recall_score(y_true, y_pred)
```

#### Python Implementation for F1 Score
F1 Score is a combined representation of both Precision and Recall.

Formula to Calculate F1 Score (F1)
>*F1 Score (F1) =  2*P*R / (P + R)*

where, P: Precision; R: Recall

```python
    #Python code for calculating
    def f1_score(y_true, y_pred):
        """
        Function to calculate f1 Score
        :param y_true: a list of true values
        :param y_pred: a list of predicted values
        :return F1 Score: F1 Score = 2PR/(P+R)

        """
        
        # Here we are using previously defined precision and
        # recall functions
        return 2*precision(y_true, y_pred)*recall(y_true, y_pred)\
                /(precision(y_true, y_pred) + recall(y_true, y_pred))
```
#### Scikit-Learn implementation of F1 Score
```python
    # Scikit-Learn implementation of F1 Score
    from sklearn.metrics import f1_score
    f1_score(y_true, y_pred)
```
### Area Under the ROC curve
Before understanding Area Under the Receiver Operating Characteristcs (ROC) cure, we need to know two terms, *True Positive Rate* and *False Positive Rate*.

- **True Positive Rate (TPR)** measures the proportion of actual positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition). True Positive Rate is same as Recall and is also called **Sensitivity**.

Formula for True Positive Rate (TPR):
>*True Positive Rate (TPR) = TP / (TP + FN)*

where, TP: True Positive; FN: False Negative

##### Python Implementation for TPR
It is same as Recall.
```python
    #Python code for calculating TPR
    def tpr(y_true, y_pred):
        """
        Function to calculate TPR of model
        :param y_true: list of true values
        :param y_pred: list of predicted values
            
        :return: TPR = TP /(TP + FN)
            
        """
        
        # here we are using the previously defined functions for
        # true_positive and false_negative
        return true_positive(y_true, y_pred)/(true_positive(y_true, y_pred)\
                                            + false_negative(y_true, y_pred))
```

-**False Positive Rate (FPR)** is calculated as the ratio between the number of negative events wrongly predicted/classified as positive (False Positives) and the total number of actual negative events (False Postive + True Negative).

Formula for False Positive Rate (FPR):
>*False Positive Rate (FPR) = FP / (FP + TN)*

where, FP: False Positive; TN: True Negative

##### Python Implementation for FPR
```python
    #Python code for calculating FPR
    def fpr(y_true, y_pred):
        """
        Function to calculate FPR of model
        :param y_true: list of true values
        :param y_pred: list of predicted values
            
        :return: FPR = FP /(FP + TN)
            
        """
        
        # here we are using the previously defined functions for
        # false_positive and true_negative
        return false_positive(y_true, y_pred)/(false_positive(y_true, y_pred)\
                                            + true_negative(y_true, y_pred)
```
-**True Negative Rate (TNR)** measures the proportion of negatives that are correctly identified as such (for example, the percentage of healthy people who are correctly identified as not having the condition). True Negative Rate (TNR) is also known as **Specificity**.

Formula for True Negative Rate (TNR)/ Specificity.
>*True Negative Rate (TNR) = TN / TN + FP*

where, TN: True Negative; FP: False Positive

Alternate formula:
>*True Negative Rate (TNR) = 1 - FPR*

where, FPR: False Positive Rate

##### Python Implementation for TNR
```python
    #Python code for calculating FPR
    def tnr(y_true, y_pred):
        """
        Function to calculate TNR of model
        :param y_true: list of true values
        :param y_pred: list of predicted values
            
        :return: TNR = TN /(TN + FP)
            
        """
        
        # here we are using the previously defined functions for
        # true_negative and false_positive
        return true_negative(y_true, y_pred)/(true_negative(y_true, y_pred)\
                                            + false_positive(y_true, y_pred)
```

Now that we know what TPR and FPR are, we can now implement Area Under ROC or AUC score. To get the ROC curve we simply just plot a FPR vs TPR graph using *matplotlib.pyplot*. and the area under the ROC is what is known as AUC. Now, lets implement it in python.

```python
    #Python code for plotting ROC curve
    
    #import matplotlib.pyplot for plotting
    import matplotlib.pyplot as plt
    tpr_list = []
    fpr_list = []

    #actual targets
    y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]

    #predicted probabilites of a sample being 1.
    y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

    #thresholds
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

    for thres in thresholds:
        temp_pred = [1 if i>= thres else 0 for i in y_pred]
        
        #calculate tpr using tpr function
        temp_tpr = tpr(y_true, temp_pred)
        #calculate fpr using fpr function
        temp_fpr = fpr(y_true, temp_pred)
        
        #append to the lists
        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)

    #plotting the ROC curve
    plt.figure(figsize= (4,3))
    plt.plot(fpr_list, tpr_list)
    plt.fill_between(fpr_list,tpr_list, alpha = 0.4)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Receiver Operating Curve")

```
#### Output of the above code
![Alt text](./images/roc_curve.png?raw=true "roc_curve")

#### Scikit-learn implementation to get AUC score
```python
    # using roc_auc_score for calculating AUC score
    # from sklearn.metrics
    from sklearn.metrics import roc_auc_score
    y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

    # calling the roc_auc_score with y_true 
    # and y_pred as arguments
    roc_auc_score(y_true, y_pred)
```
```
    # Output of above code
    0.8300000000000001
```

#### Interpreting AUC score
AUC values range from 0 to 1.
- **AUC = 1** implies you have a perfect model. Most of the time, it means that you made some mistake with validation and should revisit data processing and validation pipeline of yours. If you didn’t make any mistakes, then congratulations, you have the best model one can have for the dataset you built it on.
- **AUC = 0** implies that your model is very bad (or very good!). Try inverting the probabilities for the predictions, for example, if your probability for the positive class is p, try substituting it with 1-p. This kind of AUC may also mean that there is some problem with your validation or data processing.
- **AUC = 0.5** implies that your predictions are random. So, for any binary classification problem, if I predict all targets as 0.5, I will get an AUC of 0.5.


### Log Loss
Log loss is one of the most important evaluation matrics, because log loss penalizes quite high for an incorrect or a far-off prediction, i.e., log loss punishes you for being very sure and very wrong.

In case of binary classification problem, we define log loss as:
>*Log Loss = - 1.0 * ( target * log(prediction) + (1 - target) * log(1 - prediction) )*

Where target is either 0 or 1 and prediction is a probability of a sample belonging to class 1.

### Multi-class Log loss
For multiple samples in the dataset, the log-loss over all samples is a mere average of all individual log losses. 
>*multi-class Log Loss = - summation( target * log(prediction) + (1 - target) * log(1 - prediction) ) / n*

#### Pyhton implementation for log-loss
```python
    # python implementation for  log loss
    import numpy as np

    def log_loss(y_true, y_proba):
        """
        Function to calculate log-loss
        :param y_true: a list of true values
        :param y_proba: a list of predicted probability values
        
        :return log-loss: log-loss = -1*(y_true*log(y_proba) + (1-y_true)*log(1-y_proba)) 
        """
        
        #epsilon is used to clip probability values
        epsilon = 1e-15
        
        loss = []
        
        #loop over all true and predicted probability values
        for yt, yp in zip(y_true, y_proba):
            
            # clipping y_pred values
            yp = np.clip(yp, epsilon, 1-epsilon)
            
            # calculating log loss
            temp_loss = -1.0*( yt*np.log(yp) +( 1-yt )*np.log( 1-yp ) )
            
            # appending loss to the list
            loss.append(temp_loss)
            
        #returning the mean value of the loss
        return np.mean(loss)
```

### Metrics for Multi-Class Classification Problems
Multi-class classification problems are those in which the target column of the data has more than two classes/categories. We deal with such problems in *One-vs-Rest* fashion.

- **Macro Averaged Precision**
- **Micro Averaged Precision**
- **Weighted Averaged Precision**
- **Macro Averaged Recall**
- **Micro Averaged Recall**
- **Weighted Averaged Recall**
- **Weighted Averaged F1 Score**

##### Implementing Macro Averaged Precision in Python
```python
    def macro_averaged_precision(y_true, y_pred):
        """
        Function to calculate macro averaged precision
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: macro precision score
        
        """
        #find number of classes by taking 
        #length of unique values in true list
        num_classes = len(set(y_true))
        
        precision = 0
        
        for class_ in range(num_classes):
            
            # make all classes negative except current class
            # this is one-vs-rest method
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            #calculate true positive for current class
            tp = true_positive(temp_true, temp_pred)
            
            #calculate false positive for current class
            fp = false_positive(temp_true, temp_pred)
            
            #calculate precision for current class
            temp_precision = tp / (tp + fp)
            
            #keep adding precision for all class
            precision += temp_precision
            
        # taking average of precision
        precision = precision/num_classes
            
        return precision
```

#### Implementing Micro Averaged Precision in Python
```python
    def micro_averaged_precision(y_true, y_pred):
        """
        Function to calculate micro averaged precision
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: micro precision score        
        """
        # initialize tp as 0
        tp = 0
        # initialize fp as 0
        fp = 0
        
        # number of classes
        num_classes = len(set(y_true))
        
        for class_ in range(num_classes):
            
            # making every class negative except the current class
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            # calculating tp and fp
            # and adding
            tp += true_positive(temp_true, temp_pred)
            fp += false_positive(temp_true, temp_pred)
            
        # calculating precision
        precision = tp / (tp + fp)

        # return precision        
        return precision
```

#### Implementing Weighted Averaged Precision in Python
```python
    import numpy as np
    from collections import Counter

    def weighted_average_precision(y_true, y_pred):
        """
        Function to calculate weighted average precision
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: weighted average precision score 
        """
        
        num_classes = len(set(y_true))
        
        # create a {class : sample_count} dictionary
        # it looks something like this
        # {0: 20, 1: 15, 2: 21}
        # key = class and value = sample_count
        class_counts = Counter(y_true)
        
        #initializing precision as 0
        precision = 0
        
        for class_ in range(num_classes):
            
            # make all classes negative except current class
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]

            # calculate tp and fp
            tp = true_positive(temp_true, temp_pred)
            fp = false_positive(temp_true, temp_pred)
            
            temp_precision = tp / (tp + fp)
            
            # multiply precision with count of samples in class
            weighted_precision = class_counts[class_]*temp_precision
            
            # add to overall precision
            precision += weighted_precision
            
        # averaging the precision
        overall_precision = precision / len(y_true)
        
        # returning the averaged precision
        return overall_precision
```
#### Scikit-Learn implementation of Macro, Micro and Weighted average precision
Scikit-learn implementation of averaged precision scores is very easy, you just have to use the "precision_score()" method and change the "average" parameter of this method to "macro", "micro" or "weighted".

Below is the code for an easy implementation of Macro, Micro and Weighted average precision using scikit learn.

```python
    # scikit-learn implementation
    from sklearn.metrics import precision_score

    # Macro Averaged Precision
    macro_averaged_precision = precision_score(y_true, y_pred, average = "macro")

    # Micro Averaged Precision
    micro_averaged_precision = precision_score(y_true, y_pred, average = "micro")

    # Weighted Averaged Precision
    weighted_averaged_precision = precision_score(y_true, y_pred, average = "weighted")
```

#### Implementation of Macro, Micro and Weighted Recall in Python
>The python implementation for the averaged recalls is same as averaged precisions we saw above, with just one difference, instead of precision we calculate recall.

#### Scikit-Learn implementation of Macro, Micro and Weighted average Recall
Scikit-learn implementation of averaged recall scores is also very easy, you just have to use the "recall_score()" method and change the "average" parameter of this method to "macro", "micro" or "weighted".

Below is the code for an easy implementation of Macro, Micro and Weighted average Recall using scikit learn.

```python
    # scikit-learn implementation
    from sklearn.metrics import recall_score

    # Macro Averaged Recall
    macro_averaged_recall = recall_score(y_true, y_pred, average = "macro")

    # Micro Averaged Recall
    micro_averaged_recall = recall_score(y_true, y_pred, average = "micro")

    # Weighted Averaged Recall
    weighted_averaged_recall = recall_score(y_true, y_pred, average = "weighted")
```

#### Implementation of Weighted F1 Score in Python

```python
    import numpy as np
    from collections import Counter

    def weighted_average_f1score(y_true, y_pred):
        """
        Function to calculate weighted average f1score
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: weighted average f1 score 
        """

        # number of classes
        num_classes = len(set(y_true))

        # create class count dictionary
        # {0:12, 1:20, 2: 13}
        class_count = Counter(y_true)
        
        # initialize f1_score at 0
        f1_score = 0
        
        for class_ in range(num_classes):
            
            # make all classes negative except the current class
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            # calculating Recall
            r = recall(temp_true, temp_pred)
            
            # calculating Precision
            p = precision(temp_true, temp_pred)
            
            # checking sum of precision and recall
            if p+r != 0:
                temp_f1 = 2 * p * r/(p + r)
            else:
                temp_f1 = 0

            # giving f1 score weight
            # multiplying class count to each temp f1 score
            weighted_f1score = class_count[class_]*temp_f1
            
            # adding up weigthed f1 score
            f1_score += weighted_f1score
            
        # taking average of weighted f1 scores
        overall_f1score = f1_score/ len(y_true)

        # returning overall f1score
        return overall_f1score
```

### Confusion Matrix

A confusion matrix is nothing but a table of TP, FP, TN, and FN. With the help of confusion matrix you can quickly see how many samples were miss classified and how many were classified correctly.

```python
    # Code to display a confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true = [0,1,0,1,1,0,0,0]
    y_pred = [1,1,0,1,1,1,0,0]

    cm = confusion_matrix(y_true= y_true, y_pred= y_pred)
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
```
Output of the above code:
![Alt text](./images/confusion_matrix.png?raw=true "confusion_matrix")

NOTE: the confusion matrix we just plot is transpose of the confusion matrix that is plotted using "plot_confusion_matrix" from "sklearn.metrics" module.

### Evaluation metrics for Multi-label classification problems
**Multi-label Classification Problem**: In multiclass classification problem each sample can have one or more classes associated with it.
>For Example: You are given a dataset of images with multiple objects in it like, chair, window, or a flower pot etc., These objects are the labels/classes that are associated to the image which is one sample. In simple words, one image can have multiple targets associated with it. Such a problem is called multi-label classification problem.

The metrics for this type of classification problem are as follows:
- Precision at k (P@k)
- Average Precision at k (AP@k)
- Mean Average Precision at k (MAP@k)
- Log loss

**Precision at k (P@k)**: One must not confuse this precision with the precision discussed earlier. If you have a list of original classes for a given sample and list of predicted classes for the same, precision is defined as the number of hits in the predicted list considering only top-k predictions, divided by k.

The implementation below will make it clear:
#### Implementation of Precision at k (P@k) in Python
```python
    #precision at k

    def p_at_k(y_true, y_pred, k):
        """
        Function for calculating precision at k
        :param y_true: list of values , Actual Classes
        :param y_pred: list of values, Predicted Classes
        :return: precision at a given value k
        
        """
        #if k is 0 return 0
        #k should always be >= 1
        
        if k == 0:
            return 0
        
        #we are interested in only top-k predictions
        y_pred = y_pred[:k]
        
        #convert predictions to set
        true_set = set(y_true)
        pred_set = set(y_pred)
        
        #find common values
        common_values = true_set.intersection(pred_set)
        
        return len(common_values)/len(y_pred)
```

**Average Precision at k (AP@k)**:  AP@k is calculated using P@k. For example, if we have to calculate AP@3, we calculate P@1, P@2 and P@3 and then divide the sum by 3.

#### Implementation of Average Precision at k (AP@k) in Python
```python
    def ap_at_k(y_true, y_pred, k):
        """
        Function for calculating precision at k
        :param y_true: list of values , Actual Classes
        :param y_pred: list of values, Predicted Classes
        :return: average precision at a given value k
        
        """
        # initialize an empty list
        # for storing precisions at k (0,1,2,3,...k)
        pk_values = []
        
        for i in range(1, k+1):
            #calculate p@i and append to list
            pk_values.append(p_at_k(y_true, y_pred, i))
            
        if len(pk_values) == 0:
            return 0
        
        else:
            #return avearge precision at k
            return sum(pk_values) / len(pk_values)
        
```
**Mean Average Precision at k (MAP@k)**: MAP@k is just an average of AP@k, For example, if we have to calculate MAP@3, we calculate AP@1, AP@2 and AP@3 and then divide the sum by 3.

#### Implementation of Mean Average Precision at k (MAP@k) in Python
```python
    def map_at_k(y_true, y_pred, k):
        """
        This function calculates mean avg precision at k 
        for a single sample
        :param y_true: list of values, actual classes
        :param y_pred: list of values, predicted classes
        :return: mean avg precision at a given value k
        """
        
        # initialize an empty list
        # for storing AP@k for each k (k = 0,1,2,3..k )
        apk_values = []
        
        for i in range(len(y_true)):

            #store apk values for each sample
            apk_values.append(ap_at_k(y_true[i], y_pred[i], k=k))
            
        # return the average of AP@k_values
        return sum(apk_values)/len(apk_values)
```
### Evaluation Metrics for Regression Problems
The most common metric in regression is **Error**. It is very simple and easy to understand.

Formula for Error:
>*Error = True Value - Predicted Value*

then comes **Absolute Error** which is nothing but absolute of the error.

Formula for Absolute Error:
>*Error = |True Value - Predicted Value|*

Now that we know what *Error* and *Absolute Error* are we can discuss the other metrics, which are as follows:
- **Mean Absolute Error (MAE)**: We just simply take the mean of absolute error.
- **Mean Squared Error (MSE)**: We square the error terms and take mean.
- **Root Mean Squared Error (RMSE)**: We just simply take the square root of Mean Squared Error.
- **Squared Logarithmic Error (SLE)**: We take the log of the true and predicted values and calculate the error, and then square it.
- **Mean Squared Logarithmic Error (MSLE)**: We simply take the mean of the Squared Logarithmic Error.
- **Root Mean Squared Logarithmic Error (RMSLE)**: We simply take the root of the Mean Squared Logarithmic Error.
- **Percentage Error**: *((True Value - Predicted Value) / True Value) * 100*
- **Mean Precentage Error (MPE)**: We just take the mean of the percentage error.
- **Mean Absolute Percentage Error (MAPE)**: We take the absolute of the error and calculate percentage of error and then take the mean.
- **R^2 (R-Squared)**: Also known as **Coefficient of determination**. It says how good your model fits the data. R-squared closert to 1.0 says that the model fits the data quite well, Whereas closer to 0 means model isn't that good. It can also be negative when the model makes absurd predictions.

Now, lets see the implementation of these metrics.
### Mean Absolute Error (MAE)
Formula for MAE:
>*MAE = (1/N) * Absolute Error*

#### Python implementation for MAE:
```python
    def mae(y_true, y_pred) -> float:
        """
        Function to calculate mean absolute error
        :param y_true: List of true values (real numbers)
        :param y_pred: List of predicted values (real numbers)
        :return: mean absolute error
        """
        #initialize error at zero
        error = 0
        
        #loop over all samples in true and predicted list
        for yt, yp in zip(y_true, y_pred):
            
            #calculate absolute error
            #keep adding each error
            error += abs(yt - yp)
            
        return error/len(y_true)
```
### Mean Squared Error (MSE)
Formula for MSE:
>*MSE = (1/N) * sum((True Value - Predicted)^2)*

#### Python implementation for MSE:
```python
    def mse(y_true, y_pred) -> float:
        """
        Function to calculate mean squared error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: mean squared error
        """
        
        #initialize error at zero
        sq_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate squared error
            #add them iteratively
            sq_error += (yt - yp)**2
            
        return sq_error/ len(y_true)
```
### Root Mean Squared Error (RMSE)
Formula for RMSE:
>*RMSE = SQRT( (1/N) * sum((True Value - Predicted)^2) )*

#### Python implementation for RMSE:
```python
    def rmse(y_true, y_pred) -> float:
        """
        Function to calculate root mean squared error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: root mean squared error
        """
        
        #initialize error at zero
        sq_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate root squared error
            #add them iteratively
            sq_error += (yt - yp)**2
        
        #take mean and return
        return (sq_error/ len(y_true))**(0.5)
```

### Squared Logarithmic Error (SLE)
Formula for SLE:
>*SLE = sum( (log(1 + True Value) - log(1 + Predicted))^2) )*

#### Python implementation for SLE:
```python
    def sle(y_true, y_pred) -> float:
        """
        Function to calculate squared logarithmic error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: squared logarithmic error
        """
        
        #initialize error at zero
        sl_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate squared log error
            #add them iteratively
            sl_error += (np.log(1 + yt) - np.log(1 + yp))**2
            
        return sl_error
```

### Mean Squared Logarithmic Error (MSLE)
Formula for MSLE:
>*MSLE = (1/N) * sum((log(1 + True Value) - log(1 + Predicted))^2))*

#### Python implementation for MSLE:
```python
    def msle(y_true, y_pred) -> float:
        """
        Function to calculate mean squared logarithmic error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: mean squared logarithmic error
        """
        
        #initialize error at zero
        sl_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate squared log error
            #add them iteratively
            sl_error += (np.log(1 + yt) - np.log(1 + yp))**2
            
        #take mean of error and return    
        return sl_error/len(y_true)
```

### Root Mean Squared Logarithmic Error (RMSLE)
Formula for RMSLE:
>*RMSLE = SQRT((1/N) * sum((log(1 + True Value) - log(1 + Predicted))^2)))*

#### Python implementation for RMSLE:
```python
    def rmsle(y_true, y_pred) -> float:
        """
        Function to calculate root mean squared logarithmic error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: root mean squared logarithmic error
        """
        
        #initialize error at zero
        rsl_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate root squared log error
            #add them iteratively
            rsl_error += ((np.log(1 + yt) - np.log(1 + yp))**2)**(1/2)
            
        #take mean of error and return    
        return rsl_error/len(y_true)
```

### Percentage Error (PE)
Formula for PE:
>*PE = sum((True - Predicted) / True Value)) * 100*

#### Python implementation for Percentage Error:
```python
    def percentage_error(y_true, y_pred):
        """
        Function to calculate percentage error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: percentage error
        """
        
        #initialize error at zero
        percentage_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate root squared log error
            #add them iteratively
            percentage_error += (yt - yp) / yt
            
        #take mean of error and return    
        return percentage_error * 100
```

### Mean Percentage Error (MPE)
Formula for MPE:
>*MPE = (1/N) * sum((True - Predicted) / True Value) * 100*

#### Python implementation for MPE:
```python
    def mean_percentage_error(y_true, y_pred):
        """
        Function to calculate mean percentage error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: mean percentage error
        """
        
        #initialize error at zero
        percentage_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate root squared log error
            #add them iteratively
            percentage_error += (yt - yp) / yt
            
        #take mean of error and return    
        return percentage_error * 100/len(y_true)
```

### Mean Absolute Percentage Error (MAPE)
Formula for MAPE:
>*MAPE = (1/N) * sum(|True - Predicted| / True Value) * 100*

#### Python implementation for MAPE:
```python
    def mape(y_true, y_pred):
        """
        Function to calculate mean absolute percentage error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: mean absolute percentage error
        """
        
        #initialize error at zero
        abs_percentage_error = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate root squared log error
            #add them iteratively
            abs_percentage_error += abs(yt - yp) / yt
            
        #take mean of error and return    
        return abs_percentage_error * 100 / len(y_true)
```

### R^2 (R-Squared)
Formula for R-Squared:
>*R-Squared = 1 - sum((yt - yp) ** 2) / sum(yt - mean_yt)*

#### Python implementation for MAPE:

```python
    def r_squared(y_true, y_pred):
        """
        Function to calculate R_squared error
        :param y_true: list of true values (real numbers)
        :param y_pred: list of predicted values (real numbers)
        :return: R_squared error
        
        R_squared = 1 - sum((yt - yp)**2) / sum(yt - mean_yt) 
        """
        
        #calculate
        mean_yt = np.mean(y_true)
        
        numerator = 0
        denominator = 0
        
        #loop over all values of y_true, y_pred
        for yt, yp in zip(y_true, y_pred):
            
            #calculate the numerator and denominator part
            #add them iteratively
            numerator += (yt - yp)**2
            
            denominator += (yt - mean_yt)
            
        r_sq_error = 1 - (numerator / denominator)
        return r_sq_error
```

### SCIKIT-LEARN IMPLEMENTATION FOR ALL REGRESSION METRICS
```python
    from sklearn import metrics
    # Mean Absolute Error (MAE)
    mae = metrics.mean_absolute_error(y_true, y_pred)

    # Mean Squared Error (MSE)
    mse = metrics.mean_squared_error(y_true, y_pred, squared = True)

    # Root Mean Squared Error (RMSE)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared = False)

    # Mean Squared Logarithmic Error (MSLE)
    msle = metrics.mean_squared_log_error(y_true, y_pred, squared = True)

    # Mean Squared Logarithmic Error (RMSLE)
    rmsle = metrics.mean_squared_log_error(y_true, y_pred, squared = False)

    # Mean Absolute Percentage Error (MAPE)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)

    # R-Squared Error
    r2 = metrics.r2_score(y_true, y_pred)

```
### Some Advanced Regression Metrics
### 1. Cohen's kappa
We can easily find it in sklearn.metrics module.

```python
    # Cohen Kappa score using sklearn
    from sklearn.metrics import cohen_kappa_score
```
     
### 2. Matthew's Correlation Coefficient (MCC)
MCC ranges from -1 to 1. 1 is perfect prediction, -1 is imperfect prediction, and 0 is random prediction. 

The formula for MCC is quite simple.
>MCC =  (TP * TN - FP * FN) / sqrt( (TP + FP) * (FN + TN) * (FP + TN) * (TP + FN) )

#### Python implementation for MCC:
```python
    def mcc(y_true, y_pred):
        """
        This function calculates Matthew's Correlation Coefficient
        for binary classification.
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: mcc score
        """

        # calculating tp, tn, fp, fn using
        # previously defined functions
        tp = true_positive(y_true, y_pred)
        tn = true_negative(y_true, y_pred)
        fp = false_positive(y_true, y_pred)
        fn = false_negative(y_true, y_pred)
        
        # calculating numerator
        numerator = (tp * tn) - (fp * fn)
        
        # calculating denominator
        denominator = ((tp + fp) *(fn + tn) *(fp + tn) *(tp + fn))
        
        # taking square root of denominator
        denominator = denominator ** 0.5
        
        # returning MCC
        return numerator/denominator
```

#### Scikit-learn implementation of MCC:
```python
    from sklearn import metrics
    # Matthew's Correlation Coefficient
    mcc = metrics.matthews_corrcoef(y_true, y_pred, 
```

<a name = "chapter5">
<h1>Chapter 5 - Arranging Machine Learning Projects</h1>
</a>

We should build the classification framework in such a way that most problems will become plug n’ play. Such that, you are able to train a model without making too many changes to the code, and when you improve your models, you are be able to track them using git.

### Let’s look at the structure of the files... 
Firstly, for any project that you are doing, create a new folder. For example, I am calling the project “ML_Project”.

The inside of the project folder should look something like this:
![Alt text](./images/project_arrangement.jpeg?web=raw "project arrangement")

It should look like this:

![Alt text](./images/project_folder.png?web=raw "project folder")

- *input/*: This folder consists of all the input files and data for your machine learning  project. If you are working on NLP projects, you can keep your embeddings here.  If you are working on image projects, all images go to a subfolder inside this folder.

- *src/*: We will keep all the python scripts associated with the project here. If I talk about a python script, i.e. any *.py file (python script) is stored in the *src* folder.

- *models/*: This folder keeps all the trained models.

- *notebooks/*: All jupyter notebooks (i.e. any *.ipynb file) are stored in the notebooks folder.

- *README.md*: This is a markdown file where you can describe your project and write instructions on how to train the model or to serve this in a production environment.

- *LICENSE*: This is a simple text file that consists of a license for the project, such as MIT, Apache, etc. Going into details of the licenses is beyond the scope of this book.

### Creating First Project
For example, we will take the famous MNIST dataset. The dataset can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv). 

There are two files *"mnist_train.csv"* and *"mnist_test.csv"*, the train file consists of 60,000 images with (in rows) and 785 columns (784 image pixel values) and (1 column for image label).

### Exploration Notebook
In "notebook/" folder, we create a python notebook for exploring our data. Data exploration is the first and most important step in an ML project. In this step, we explore the data, and based upon that we decide what techniques are applicable for that type of data.

#### Code to explore the data
Firstly, we import all the important libraries.
```python
    #notebooks/exploration.ipynb

    #importing all the important packages
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

```

Now, we read the data from our input folder.
```python
    # this code reads the csv file from the
    # given path
    df = pd.read_csv("../input/mnist_train.csv")
```

Plotting the countplot to see the distribution of data.
```python
    # using seaborn to plot countplot
    sns.countplot(data = df, x = "label")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.show()
```

The above code will give the following output:
![Alt text](./images/count_plot.png?web=raw "count plot")

Seeing this plot we can say there is no significant skeweness in the data. Hence, we can use Accuracy, Precision, Recall, and F1 Score as evaluation metrics.

### Create K-Fold data for Validation
The next step after deciding the metric is, creating the K-Fold data for validation. For that we create a *"create_folds.py"* script in our "src/" folder. 

And write the following code:
```python
    # src/create_folds.py

    #import necessary packages
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold

    # read the data
    train_df = pd.read_csv("..input/mnist_train.csv")
    
    # we create new column called kfold and fill it with -1
    train_df["kfold"] = -1

    # shuffling the rows
    train_df = train_df.sample(frac =1).reset_index(drop=True)

    # initialize object of KFold class
    kf = KFold(n_splits=5)

    # creating folds
    for fold, (trn_, val_) in enumerate(kf.split(X = train_df)):
        train_df.loc[val_, 'kfold'] = fold
        
    # export the data with folds to input folder
    train_df.to_csv("../input/mnist_train_kfolds.csv", index=False)

```
This will create a new file in the input/ folder called "mnist_train_folds.csv", and it’s the same as "mnist_train.csv". The only differences are that this CSV is shuffled and has a new column called kfold. 

Now that we have created the data with folds, we are good to go with creating a basic model. This is done in *"train.py", "config.py", "model_dispatcher.py"*  in "src/" folder.

#### config.py
```python
    # src/config.py

    # just store the paths in variables
    TRAINING_FILE = "../input/mnist_train_folds.csv"
    MODEL_OUTPUT = "../models/"

```

#### model_dispatcher.py
```python
    # src/model_dispatcher.py
    # dictionary of models
    from sklear.tree import DecisionTreeClassifier()

    models = {"decision_tree_gini": DecisionTreeClassifier(criterion = "gini"),
                "decision_tree_entropy": DecisionTreeClassifier(criterion = "entropy"),
                }
```

#### train.py
```python
    #src/train.py

    # import all the necessary libraries
    import pickle
    import argparse
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    #importing config.py
    import config

    #import model_dispatcher.py
    import model_dispatcher
    


    def run(fold):
        #read the training data with folds
        df = pd.read_csv(config.TRAINING_FILE)
        
        #training data is where kfold is not equal to provided fold
        #also, note that we reset the index
        
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)
        
        #drop the label column from and make x_train and y_train
        #using ".values" to convert data into numpy array
        X_train = df_train.drop("label", axis=1).values
        y_train = df_train["label"].values
        
        
        # for Validation
        # drop the label column from and make x_valid and y_valid
        # using ".values" to convert data into numpy array
        X_valid = df_valid.drop("label", axis=1).values
        y_valid = df_valid["label"].values
        
        
        # initializing model class using model_dispatcher
        clf = model_dispatcher.models[model]
        
        # fit the data to the model
        clf.fit(X=X_train, y=y_train)
        
        # predict labels for Validation data
        predicted = clf.predict(X_valid)
        
        # calculate and print accuracy
        accuracy = accuracy_score(y_valid, predicted)
        print(f"Fold = {fold}, Accuracy = {accuracy}")
        
        
        #save model in models/ folder
        with open(f"{config.MODEL_OUTPUT}dt_{fold}.bin", "wb") as f:
            pickle.dump(clf, f)
        
    if __name__ == "__main__":
        #initialize ArgumentParser class
        parser = argparse.ArgumentParser()
        
        # add the different arguments you need and their type
        parser.add_argument("--fold", type=int)
        
        parser.add_argument("--model", type=str)
        
        #read the arguments from command line
        args = parser.parse_args()
        
        #run the folds specified by commmand line arguments
        
        run(
            fold = args.fold
            mode = args.model
        )
```

When we are done writing scripts, we can run our train.py script in a terminal as follows:
```
    python train.py --fold 0 --model decision_tree_gini
    Fold=0, Accuracy=0.8665833333333334
```

We can change the folds and model by changing the commands as:

```
    python train.py --fold 1 --model decision_tree_entropy
    Fold=1, Accuracy=0.8705833333333334
```

In this way we can put any model in model_dispatcher like RandomForestClassifier, LogisticRegression etc. and train our models by passing arguments in any terminal.

<a>
<h1>Chapter 6 - Approaching Categorical Variables</h1>
</a>

In this chapter we will learn how to deal with **categorical variables** in the data.

### What are Categorical Variables?
A categorical variable is a variable (column/feature in a dataset) that has two or more than two but finite values, It contain a finite set of text values which needs to be converted to numerical values so later can be used to train a machine learning model.

### Categorical Variables are of Two Types
- **Nomial**:  Nomial variables are variables that have two or more categories which do not have any kind of order associated with them For example, if gender is classified into two groups, i.e. male and female, it can be considered as a nominal variable.

- **Ordinal**: Ordinal variables on the other hand, have “levels” or categories with a particular order associated with them. For example, a variable named temperature can take; low, medium and high as values.

- **Binary**: It is a categorical variable that takes exactly two values as categories, for example, a variable that takes Yes or No as values.

- **Cyclic**: As the name says, Cyclic type of variables have values which are *cyclic*. for example, a variable that takes values of week days, Sun, Mon, Tue, Wed, Thu, Fri, Sat, and after Saturday we have Sunday again.

### Dealing with Categorical Variables
For this, let's learn with the famous *"cat-in-the-dat"* dataset. The dataset can be found [here](https://www.kaggle.com/competitions/cat-in-the-dat-ii/data).

#### Exploring the data
Let's have a look at the data -

![Alt text](./images/cat_dat_dataset.png?web=raw "cat-in-the-dat")
This is how the data looks.


Let's see the data information.
![Alt text](./images/cat_dat_info.png?web=raw "cat-in-the-dat info")
As we can see in the image above, There are 23 categorical variables (excluding "id" and "target" columns). The data consist of all four types of categorical variables such as Nomial, Ordinal, Binary and Cyclic ("day" and "month" column).

Overall, there are:
- Five binary variables
- Ten nominal variables
- Six ordinal variables
- Two cyclic variables
- And a target variable

Now, let's plot a count plot to see the distribution of data and to decide what evaluation metric we can use.

Here is the countplot of the data:
![Alt text](./images/cat_dat_countplot.png?web=raw "cat-in-the-dat countplot")
As we can clearly see there is skewness in the data, that is the samples with target value "0" are more compared to "1". So, with the above analysis we can say we should avoid using Accuracy as the evaluation metric, we will use AUC score for this data.

### Encoding the Categorical Variables
As the data is full of categorical variables, we need to convert each variable to numerical type, and this process is called **Encoding**.

### Label Encoding
This is a very easy encoding technique. In this, we assign each category a numerical value. We can either do it manually or using scikit learn for this.

#### Manual Label Encoding
Given below is the code to manually perform the label encoding.
```python
    # encoding ord_2 column in the data
    # create a mapping
    # map each categorical value a numerical value
    mapping = {
                'Freezing':0,
                'Warm':1,
                'Cold':2,
                'Boiling Hot':3,
                'Hot':4,
                'Lava Hot':5, 
                }

    df["ord_2"] = df.ord_2.map(mapping)

```

Value counts before mapping:
```
    df.ord_2.value_counts()
    Freezing        142726
    Warm            124239
    Cold            97822
    Boiling Hot     84790
    Hot             67508
    Lava Hot        64840
    Name: ord_2, dtype: int64
```

Value counts after mapping:
```
    0.0     142726
    1.0     124239
    2.0     97822
    3.0     84790
    4.0     67508
    5.0     64840
    Name: ord_2, dtype: int64
```
#### Using LabelEncoder from Scikit-learn
```python
    import pandas as pd
    from sklearn import preprocessing

    # read the data
    df = pd.read_csv("../input/cat_train.csv")

    # fill NaN values in ord_2 column
    df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")
    
    # initialize LabelEncoder
    lbl_enc = preprocessing.LabelEncoder()
    
    # fit label encoder and transform values on ord_2 column
    # P.S: do not use this directly. fit first, then transform
    df.["ord_2"] = lbl_enc.fit_transform(df.ord_2.values)
```
NOTE: You will see that we used "fillna" from pandas. The reason is LabelEncoder from scikit-learn does not handle NaN values, and ord_2 column has NaN values in it.

Now, that we have label encoded our columns using LabelEncoder(), we can directly train any tree-based model using this data such as, Decision Tree, Random Forest or XGboost etc. but we cannot train any linear model directly with this data, for that the data needs to be **normalized** (or **standardized**).

For training linear models such as, Logistic Regression or SVM etc. We can binarize the data:

```
    Freezing    --> 0 -->   0 0 0
    Warm        --> 1 -->   0 0 1
    Cold        --> 2 -->   0 1 0
    Boiling Hot --> 3 -->   0 1 1
    Hot         --> 4 -->   1 0 0
    Lava Hot    --> 5 -->   1 0 1
```
To binarize the data, simply convert the categorical values to numerical and then convert these numerical values to their binary representation.

#### But... Binarizing the data takes a lot of memory.
To reduce the memory consumption, we can store the data in **sparse format**. In a sparse format we only keep the values that matter. In case of binary variables described above, all that matters is where we have "1s".

To understand how sparse representation works, suppose we have three features as shown in figure below:
![Alt text](./images/binary_repr.jpeg?web=raw "three features")

We convert this data to binary representation, and it looks like this:
![Alt text](./images/binary_repr2.jpeg?web=raw "binary representation")

Now let's turn this into sparse format, to represent this matrix only with ones create a dictionary in which keys are indices of rows and columns and value is 1:
```
    (0, 2)  1
    (1, 0)  1
    (2, 0)  1
    (2, 2)  1
```
The above format is sparse format which takes much less memory than binary representation of data. Any numpy array can be converted to a sparse matrix by simple python code.

```python
    # converting a numpy array into
    # a sparse matrix
    import numpy as np
    from scipy import sparse
    # create our example feature matrix
    example = np.array(
    [
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 1]
    ]
    )
    # convert numpy array to sparse CSR matrix
    sparse_example = sparse.csr_matrix(example)
```

### One Hot Encoding
One Hot Encoding is another transformation technique for categorical variables that takes even less memory than sparse format. One hot encoding is a binary encoding too in the sense that there are only two values, 0s and 1s. However, it’s not a binary representation.

Suppose we represent each category of the ord_2 variable by a vector. This vector is of the same size as the number of categories in the ord_2 variable. Each vector is of size six and has all zeros except at one position.

And it looks like this:
![Alt text](./one_hot_encode.jpeg?web=raw "one hot encode")
Each vector has a 1 and rest all other values are 0s.

Let't One-hot-encode the following data:
![Alt text](./images/binary_repr.jpeg?web=raw "three features")

After one-hot-encoding:
![Alt text](./images/one_hot_encode2.jpeg?web=raw "one hot encoded features")





