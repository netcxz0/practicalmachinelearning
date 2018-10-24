---
title: "Practical Machine Learning Course Project"
author: "Cheng Zheng"
date: "October 24, 2018"
output: 
  html_document: 
    keep_md: yes
---



## Summary
In this project, we will get the data collected from accelerometers strapped on belt, forearm, arm, and dumbell of 6 participants; use the random forest algorithm to build the prediction model; apply the trained model to the testing data set to predict what type of exercise was done.  The data for this prediction was collected by a group of researchers, Velloso et al 2013 ^[1]^.  We will use k-fold cross validation with k set to 10 for cross validation in the training data set. 


## Weight Lifting Exercises Data ^[1]^

This human activity recognition research has traditionally focused on discriminating between different activities, to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

The training datat for this project are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)



## Exploratory Data Analysis

```r
training = read.csv("pml-training.csv")
testing = read.csv("pml-testing.csv")
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

There are 159 prediction variables in both training and testing data set, and 1 dependent variable.  We will apply the learned model to the 20 test cases available in the test data set to predict 20 different test cases.  Among 159 independent variables, many of them contain the NA values, and not used for the prediction.  Let's remove the independent variables that have NA value from the training and test data set. This results in 59 independent variables in both training and testing data set.



```r
dim(training)
```

```
## [1] 19622    59
```

```r
dim(testing)
```

```
## [1] 20 59
```

## Build the prediction model

### Random Forests
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes(classification) or mean prediction of the individual trees^[2]^. Random forests correct the decision tree's habit of overfitting to their trainning set ^[3]:587-588^.

### K-fold cross validation
In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k - 1 subsamples are used as training data. The cross-validation process is then repeated k times, with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling (see below) is that all observations are used for both training and validation, and each observation is used for validation exactly once. 10-fold cross-validation is commonly used^[4]^, but in general k remains an unfixed parameter.

In the R Caret package, we can use the traincontrol function to enable the k-fold cross validation and parallel processing to speed up the training speed.


```r
library(caret)

#enable parallel processing to speed up
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() -1)  # convention to leave 1 core for OS
registerDoParallel(cluster)
```

```r
# configure trainControl object
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# use random forest with 10-fold cross validation and parallel process to train the model 
modFitRf <- train(classe ~ ., data = training, model = "rf", trControl = fitControl, ntree = 10) 

# De-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
```

### The trained Random Forest prediction model

```r
modFitRf
```

```
## Random Forest 
## 
## 19622 samples
##    58 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17661, 17660, 17660, 17659, 17658, 17661, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9769152  0.9707840
##   41    0.9991336  0.9989041
##   80    0.9984201  0.9980017
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 41.
```

```r
modFitRf$resample
```

```
##     Accuracy     Kappa Resample
## 1  0.9989806 0.9987107   Fold02
## 2  0.9994901 0.9993550   Fold01
## 3  0.9994906 0.9993557   Fold04
## 4  0.9994903 0.9993553   Fold03
## 5  0.9994901 0.9993549   Fold06
## 6  1.0000000 1.0000000   Fold05
## 7  0.9989812 0.9987113   Fold08
## 8  0.9994903 0.9993553   Fold07
## 9  0.9979602 0.9974198   Fold10
## 10 0.9979623 0.9974225   Fold09
```

```r
confusionMatrix.train(modFitRf)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.0  0.0
##          D  0.0  0.0  0.0 16.4  0.0
##          E  0.0  0.0  0.0  0.0 18.4
##                             
##  Accuracy (average) : 0.9991
```

### Prediction on the 20 test cases in the testing data

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion

Using the Random Forest algorithm with 10 fold cross validation on the training data set 0f 19,622 samples, the trained model has 99.9% accuracy on the training data set.  

We used the trained model to predict the 20 test cases in the test data set, and got the 100% accuracy.

## References
[1] Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 

[2] Ho, Tin Kam (1995). Random Decision Forests (PDF). Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal, QC, 14-16 August 1995. pp. 278-282. Archived from the original (PDF) on 17 April 2016. Retrieved 5 June 2016. 

[3] Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome (2008). The Elements of Statistical Learning (2nd ed.). Springer. ISBN 0-387-95284-5.

[4] McLachlan, Geoffrey J.; Do, Kim-Anh; Ambroise, Christophe (2004). Analyzing microarray gene expression data. Wiley.
