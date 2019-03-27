#The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
#It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 
#The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
library(data.table)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(pROC)
library(glmnet)
library(caret)
library(Rtsne)
library(xgboost)
library(doParallel)
data <- fread("creditcard.csv")
normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
}
data$Amount <- normalize(data$Amount)

#To avoid developing a “naive” model, we should make sure the classes are roughly balanced. Therefore, we will use a resampling (and, more precisely, oversampling) scheme called SMOTE. It works roughly as follows:
#The algorithm selects 2 or more similar instances of data
#It then perturbs each instance one feature at a time by a random amount. This amount is within the distance to the neighbouring examples.
# Set random seed for reproducibility
set.seed(42)
# Transform "Class" to factor to perform classification and rename levels to predict class probabilities (need to be valid R variable names)
data$Class <- as.factor(data$Class)
# Create training and testing set with stratification (i.e. preserving the proportions of false/true values from the "Class" column)
train_index <- createDataPartition(data$Class, times = 1, p = 0.8, list = F)
X_train <- data[train_index]
X_test <- data[!train_index]
y_train <- data$Class[train_index]
y_test <- data$Class[-train_index]
# Parallel processing for faster training
registerDoMC(cores = 8)
# Use 10-fold cross-validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",
                     summaryFunction = twoClassSummary,
                     savePredictions = T)
#Logistic regression is a simple regression model whose output is a score between 0 and 1.
#The model can be fitted using gradient descent on the parameter vector beta. Equipped with some basic information, let’s fit the model and see how it performs!
log_mod <- glm(Class ~ ., family = "binomial", data = X_train)
summary(log_mod)

# Use a threshold of 0.5 to transform predictions to binary
y_predict <- as.numeric(predict(log_mod, X_test, type = "response") > 0.5)
y_test <- as.factor(y_test)
y_predict <- as.factor(y_predict)
conf_mat <- confusionMatrix(y_test, y_predict )
print(conf_mat)
fourfoldplot(conf_mat$table)
#A simple logistic regression model achieved nearly 100% accuracy, with ~99% precision (positive predictive value) and ~100% recall (sensitivity). We can see there are only 6 false negatives (transactions which were fraudulent in reality but ont identified as such by the model). This means that the baseline model will be very hard to beat.
y_predict2 <- as.numeric(predict(log_mod, X_test, type = "response") > 0.999)
y_test <- as.factor(y_test)
y_predict2 <- as.factor(y_predict2)
#We can further minimise the number of false negatives by increasing the classification threshold. However, this comes at the expense of identifying some legitiate transactions as fraudulent. This is typically of much lesser concern to banks and it is the false negative rate that should be minimized
conf_mat2 <- confusionMatrix(y_test, y_predict2 )
print(conf_mat2)
#Now we have just 2 false negatives, but we identified many more legitimate transactions (72) as fraudulent compared to 0.5 threshold. When adjusting the classification threshold, we can have a look at the ROC curve to guide us.
roc_logmod <- roc(y_test, as.numeric(predict(log_mod, X_test, type = "response")))
# Train a Random Forest classifier, maximising recall (sensitivity)
X_train_rf <- X_train
X_train_rf$Class <- as.factor(X_train_rf$Class)
levels(X_train_rf$Class) <- make.names(c(0, 1))
model_rf_smote <- train(Class ~ ., data = X_train_rf, method = "rf", trControl = ctrl, verbose = T, metric = "ROC")
model_rf_smote
#It is important to note that SMOTE resampling was done only on the training data. The reason for that is if we performed it on the whole dataset and then made the split, SMOTE would bleed some information into the testing set, thereby biasing the results in an optimistic way.
#The results on the training set look very promising. 

preds <- predict(model_rf_smote, X_test, type = "prob")
predict3 <- as.factor(preds$X1 > 0.5)

y_predict3 <- as.numeric(preds$X1 > 0.5)
y_test <- as.factor(y_test)
y_predict3 <- as.factor(y_predict3)
conf_mat3 <- confusionMatrix( y_predict3, y_test)
print(conf_mat3)
#we can also try XGBoost, which is based on Gradient Boosted Trees and is a more powerful model compared to both Logistic Regression and Random Forest.
dtrain_X <- xgb.DMatrix(data = as.matrix(X_train[,-c("Class")]), label = as.numeric(X_train$Class))
dtest_X <- xgb.DMatrix(data = as.matrix(X_test[,-c("Class")]), label = as.numeric(X_test$Class))
xgb <- xgboost(data = dtrain_X, nrounds = 100, gamma = 0.1, max_depth = 10, objective = "binary:logistic", nthread = 7)
preds_xgb <- predict(xgb, dtest_X)
y_predict4 <- as.factor(preds_xgb > 0.5)
y_test <- as.factor(y_test)
conf_mat4 <- confusionMatrix(as.numeric(y_predict4, y_test))
print(conf_mat4)
#We can see the model performs much better than the previous ones, espeically in terms of Negative Predictive Value, while still achieving nearly ~100% precision and recall on the validation set! Once again, we can set the classification threshold using the ROC curve.
roc_xgb <- roc(y_test, preds_xgb)
plot(roc_xgb, main = paste0("AUC: ", round(pROC::auc(roc_xgb), 3)))
#This project has explored the task of identifying fraudlent transactions based on a dataset of anonymised features. It has been shown that even a very simple logistic regression model can achieve good recall, while a much more complex Random Forest model improves upon logistic regression in terms of AUC. However, XGBoost model improves upon both models.
