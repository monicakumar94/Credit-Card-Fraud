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

# Set random seed for reproducibility
set.seed(42)
# Transform "Class" to factor to perform classification and rename levels to predict class probabilities (need to be valid R variable names)
data$Class <- as.numeric(data$Class)
#data$Class <- revalue(data$Class, c("0"="false", "1"="true"))
#data$Class <- factor(data$Class, levels(data$Class)[c(2, 1)])
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

log_mod <- glm(Class ~ ., family = "binomial", data = X_train)
summary(log_mod)
y_predict <- as.numeric(predict(log_mod, X_test, type = "response") > 0.5)

y_test <- as.factor(y_test)
y_predict <- as.factor(y_predict)
conf_mat <- confusionMatrix(y_test, y_predict )
print(conf_mat)
fourfoldplot(conf_mat$table)
y_predict2 <- as.numeric(predict(log_mod, X_test, type = "response") > 0.999)
y_test <- as.factor(y_test)
y_predict2 <- as.factor(y_predict2)
conf_mat2 <- confusionMatrix(y_test, y_predict2 )
print(conf_mat2)
roc_logmod <- roc(y_test, as.numeric(predict(log_mod, X_test, type = "response")))
X_train_rf <- X_train
X_train_rf$Class <- as.factor(X_train_rf$Class)
levels(X_train_rf$Class) <- make.names(c(0, 1))
model_rf_smote <- train(Class ~ ., data = X_train_rf, method = "rf", trControl = ctrl, verbose = T, metric = "ROC")
model_rf_smote
preds <- predict(model_rf_smote, X_test, type = "prob")
predict3 <- as.factor(preds$X1 > 0.5)

y_predict3 <- as.numeric(preds$X1 > 0.5)
y_test <- as.factor(y_test)
y_predict3 <- as.factor(y_predict3)
conf_mat3 <- confusionMatrix( y_predict3, y_test)
print(conf_mat3)
dtrain_X <- xgb.DMatrix(data = as.matrix(X_train[,-c("Class")]), label = as.numeric(X_train$Class))
dtest_X <- xgb.DMatrix(data = as.matrix(X_test[,-c("Class")]), label = as.numeric(X_test$Class))
xgb <- xgboost(data = dtrain_X, nrounds = 100, gamma = 0.1, max_depth = 10, objective = "binary:logistic", nthread = 7)
preds_xgb <- predict(xgb, dtest_X)
y_predict4 <- as.factor(preds_xgb > 0.5)
y_test <- as.factor(y_test)
conf_mat4 <- confusionMatrix(as.numeric(y_predict4, y_test))
print(conf_mat4)