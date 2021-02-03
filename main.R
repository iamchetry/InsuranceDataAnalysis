#setwd('../iamchetry/Documents/UB_files/506/hw_3/')

#install.packages('DMwR')
#install.packages("pROC")
#install.packages("devtools")
#install.packages("ggfortify")

#---------------- Load Libraries ------------------
library(Metrics)
library(DMwR)
library(caret)
library(leaps)
library(comprehenr)
library(glue)
library(pROC)
library(pls)
library(glmnet)
library(devtools)
library(ggfortify)
library(ggplot2)


#------------------------ 1st Question --------------------------
set.seed(3)
train_ = read.table('ticdata2000.txt')
x = to_vec(for (i in 1:85) glue('x_{i}_'))
names(train_) = c(x, 'y')
dim(train_)
head(train_)

test_ = read.table('ticeval2000.txt')
head(test_)
names(test_) = x
head(test_)

y_test = read.table('tictgts2000.txt')
y_test = as.factor(y_test$V1)

train_$x_1_ = as.factor(train_$x_1_)
train_$x_4_ = as.factor(train_$x_4_)
train_$x_5_ = as.factor(train_$x_5_)
train_$x_6_ = as.factor(train_$x_6_)
train_$x_44_ = as.factor(train_$x_44_)
train_$y = as.factor(train_$y)

test_$x_1_ = as.factor(test_$x_1_)
test_$x_4_ = as.factor(test_$x_4_)
test_$x_5_ = as.factor(test_$x_5_)
test_$x_6_ = as.factor(test_$x_6_)
test_$x_44_ = as.factor(test_$x_44_)

table(train_$y)
new_train = SMOTE(y~., train_, perc.over = 1400, perc.under = 113)
table(new_train$y)

#--------- Linear Model ----------
new_train$y = as.factor(new_train$y)
model_lm = glm(y~., data = new_train, family='binomial')
summary(model_lm)
train_preds = predict(model_lm, type = 'response')

#Finding optimal threshold to calculate decision boundary
analysis = roc(response=new_train$y, predictor=train_preds)
e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
opt_t = subset(e,e[,2]==max(e[,2]))[,1]

test_preds = predict(model_lm, newdata = test_, type = 'response')

par(mfrow = c(2,1))
hist(train_preds, xlab = 'Predicted Probabilities', ylab = 'Counts',
     main = 'Distribution of Train Probabilities')
hist(test_preds, xlab = 'Predicted Probabilities', ylab = 'Counts',
     main = 'Distribution of Test Probabilities')

train_preds = ifelse(train_preds >= opt_t, '1', '0') # Train Prediction
hist(train_preds)

test_preds = ifelse(test_preds >= opt_t, '1', '0') # Test Prediction

train_actual = new_train$y

tab_train = table(train_preds, train_actual)
tab_test = table(test_preds, y_test)

#--------- Confusion Matrix to determine Accuracy ---------
conf_train = confusionMatrix(tab_train) 
conf_test = confusionMatrix(tab_test)

train_error = 1 - round(conf_train$overall['Accuracy'], 4) # Training Error
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

print(train_error)
print(test_error)

#-------------- Subset Selection ---------------
new_train$y = as.factor(new_train$y)

bw_sub = regsubsets(y~., data = new_train, nbest = 1, nvmax = 147,
                    method = "backward") # Backward SS
bw_summary = summary(bw_sub)

par(mfrow = c(2,1))
plot(bw_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l",
     main='For Exhaustive Subset Selection')
plot(bw_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2",
     type = "l")

forward_sub = regsubsets(y~., data = new_train, nbest = 1, nvmax = 147,
                         method = "forward") #Forward SS
forward_summary = summary(forward_sub)

par(mfrow = c(2,1))
plot(forward_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l",
     main='For Forward Subset Selection')
plot(forward_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2",
     type = "l")

# Create dummies for Train data
temp_train = cbind(rep(1, length(new_train[, 1])), new_train) # Creating Intercept Column
names(temp_train) = c('(Intercept)', names(new_train)) 
train_dummy = model.matrix(~x_1_+x_4_+x_5_+x_6_+x_44_, data = temp_train)
train_dummy = subset(train_dummy, select = -c(1))
temp_train = cbind(temp_train, train_dummy)

# Create dummies for Test data
temp_test = cbind(rep(1, length(test_[, 1])), test_) # Creating Intercept Column
names(temp_test) = c('(Intercept)', names(test_)) 
test_dummy = model.matrix(~x_1_+x_4_+x_5_+x_6_+x_44_, data = temp_test)
test_dummy = subset(test_dummy, select = -c(1))
temp_test = cbind(temp_test, test_dummy)

bw_mse_list = list() # List created to store MSE for each Subset of Backward SS

for (i in 1:147)
{
  coeff_ = coef(bw_sub, id=i)
  abs_ = setdiff(c(names(coeff_)), c(names(temp_test)))
  for (x in abs_){
    temp_test[c(x)] = 0           # Filling missing columns
  }
  d = temp_test[c(names(coeff_))]
  test_preds = t(coeff_%*%t(d))
  
  d = temp_train[c(names(coeff_))]
  train_preds = t(coeff_%*%t(d))
  
  par(mfrow = c(2,1))
  hist(train_preds, xlab = 'Predicted Probabilities', ylab = 'Counts',
       main = 'Distribution of Train Probabilities')
  hist(test_preds, xlab = 'Predicted Probabilities', ylab = 'Counts',
       main = 'Distribution of Test Probabilities')
  
  #Finding optimal threshold to calculate decision boundary
  analysis = roc(response=new_train$y, predictor=train_preds)
  e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
  opt_t = subset(e,e[,2]==max(e[,2]))[,1]

  test_preds = ifelse(test_preds >= opt_t, '1', '0') # Test Prediction
  tab_test = table(test_preds, y_test)
  
  conf_test = confusionMatrix(tab_test)
  test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

  bw_mse_list[[glue('{i}')]] = test_error
}

forward_mse_list = list() # List created to store MSE for each Subset of Forward SS

for (i in 1:147)
{
  coeff_ = coef(forward_sub, id=i)
  d = temp_test[c(names(coeff_))]
  test_preds = t(coeff_%*%t(d))
  
  d = temp_train[c(names(coeff_))]
  train_preds = t(coeff_%*%t(d))

  #Finding optimal threshold to calculate decision boundary
  analysis = roc(response=new_train$y, predictor=train_preds)
  e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
  opt_t = subset(e,e[,2]==max(e[,2]))[,1]

  test_preds = ifelse(test_preds >= opt_t, '1', '0') # Test Prediction
  tab_test = table(test_preds, y_test)
  
  conf_test = confusionMatrix(tab_test)
  test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error
  
  forward_mse_list[[glue('{i}')]] = test_error
}

par(mfrow = c(2,1))
plot(unlist(bw_mse_list), main = 'Test Performance : Backward Subset Selection',
     xlab = 'No. of Variables', ylab = 'Test MSE', col='blue')
plot(unlist(forward_mse_list), main = 'Test Performance : Forward Subset Selection',
     xlab = 'No. of Variables', ylab = 'Test MSE', col='red')


#------------ X and Y Splitting -------------
X = model.matrix(~., subset(new_train, select = -c(86)))
y = as.numeric(new_train$y)

missing = setdiff(c(colnames(X)), c(colnames(temp_test))) # Missing Column
for (x in missing){
  temp_test[c(x)] = 0           # Filling missing columns
}

#------------ Ridge Regression --------------
ridge_model = glmnet(X, y, alpha = 0)
ridge_out = cv.glmnet(X, y, alpha = 0)

bestlam = ridge_out$lambda.min

actual_predictions = predict(ridge_model, s=bestlam, newx = X, type = "response")

#Finding optimal threshold to calculate decision boundary
analysis = roc(response=new_train$y, predictor=actual_predictions)
e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
opt_t = subset(e,e[,2]==max(e[,2]))[,1]

ridge_predictions = predict(ridge_model, s=bestlam, 
                            newx=as.matrix(temp_test[, c(colnames(X))]),
                            type = "response")

ridge_predictions = ifelse(ridge_predictions >= opt_t, '1', '0') # Test Prediction
tab_test = table(ridge_predictions, y_test)

conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error
print(test_error)


#-------------- LASSO Regression ---------------
lasso_model = glmnet(X, y, alpha = 1)
lasso_out = cv.glmnet(X, y, alpha = 1)

bestlam = lasso_out$lambda.min

actual_predictions = predict(lasso_model, s=bestlam, newx=X, type = "response")

#Finding optimal threshold to calculate decision boundary
analysis = roc(response=new_train$y, predictor=actual_predictions)
e = cbind(analysis$thresholds, analysis$sensitivities+analysis$specificities)
opt_t = subset(e,e[,2]==max(e[,2]))[,1]

lasso_predictions = predict(lasso_model, s=bestlam, 
                            newx=as.matrix(temp_test[, c(colnames(X))]),
                            type = "response")

lasso_predictions = ifelse(lasso_predictions >= opt_t, '1', '0') # Test Prediction
tab_test = table(lasso_predictions, y_test)

conf_test = confusionMatrix(tab_test)
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error
print(test_error)

par(mfrow = c(2,1))
plot(ridge_out, main='Ridge')
plot(lasso_out, main='LASSO')


#--------------------- 2nd Question -----------------------
set.seed(2)
data_ = as.data.frame(cbind(matrix(rnorm(1000*10,mean=0,sd=1), 1000, 10), 
                            matrix(runif(1000*10), 1000, 10)))
coeffs_ = runif(16, min = -1, max=1)
data_coeffs = data_[-c(9, 10, 11, 12)] # V9, 10, 11 and 12 set to 0
e = runif(1000, min = -0.1, max=0.1)
y = as.matrix(data_coeffs)%*%as.matrix(coeffs_) + e # Y=B*X+e

data_ = cbind(data_, y)
attach(data_)

t = createDataPartition(y, p=0.8, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])

best_sub = regsubsets(y~., data = train_, nbest = 1, nvmax = 20,
                      method = "exhaustive") # Backward SS
best_summary = summary(best_sub)

par(mfrow = c(2,1))
plot(best_summary$rss, xlab = "Number of Variables", ylab = "Training RSS",
     type = "l", main='For Exhaustive Subset Selection')
plot(best_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2",
     type = "l")

y_test = test_$y # Actual Ratings
temp_test = cbind(rep(1, length(test_[, 1])), test_) # Creating Intercept Column
names(temp_test) = c('(Intercept)', names(test_)) 

best_mse_list = list() # List created to store MSE for each Subset of Exhaustive SS

for (i in 1:20)
{
  coeff_ = coef(best_sub, id=i)
  d = temp_test[c(names(coeff_))]
  predicted_y = t(coeff_%*%t(d))
  
  e = mse(as.numeric(unlist(y_test)), as.numeric(predicted_y))
  best_mse_list[[glue('{i}')]] = e
}

plot(unlist(best_mse_list), main = 'Test Performance : Exhaustive Subset Selection',
     xlab = 'No. of Variables', ylab = 'Test MSE', col='red')

coeff_ = coef(best_sub, id=16)


#------------------- 3rd Question ---------------------
data_ = iris
attach(data_)
par(mfrow = c(1, 1))
boxplot(Sepal.Length, horizontal = TRUE)
boxplot(Sepal.Width, horizontal = TRUE, main='Boxplot for Sepal Width')
boxplot(Petal.Length, horizontal = TRUE)
boxplot(Petal.Width, horizontal = TRUE)

data_ = subset(data_, Sepal.Width<=4 & Sepal.Width!=2) # Outlier Removal

#--------- Stratified Samping into Train and Test based on Species variable ---------
t = createDataPartition(Species, p=0.65, list = FALSE)
train_ = na.omit(data_[t, ])
test_ = na.omit(data_[-t, ])

#---------------------- K-Nearest Neighbor ------------------------
error_list_train = list() # List created to store Training Error for each value of K
error_list_test = list() # List created to store Testing Error for each value of K

require(class)
for (k in seq(1, 61, 2))
{
  KNN_train = knn(train_[-c(5)], train_[-c(5)], train_$Species, k) # Train Prediction
  KNN_test = knn(train_[-c(5)], test_[-c(5)], train_$Species, k) # Test Prediction
  
  train_predicted = as.factor(KNN_train)
  test_predicted = as.factor(KNN_test)
  
  train_actual = as.factor(train_$Species)
  test_actual = as.factor(test_$Species)
  
  tab_train = table(train_predicted, train_actual)
  tab_test = table(test_predicted, test_actual)
  
  #-------- Confusion Matrix for Accuracy ---------
  conf_train = confusionMatrix(tab_train)
  conf_test = confusionMatrix(tab_test)
  
  error_list_train[[glue('{k}')]] = 1 - round(
    conf_train$overall['Accuracy'], 4)
  error_list_test[[glue('{k}')]] = 1 - round(
    conf_test$overall['Accuracy'], 4)
  
}

#------------ Plotting Errors for all values of K ---------------
par(mfrow = c(2,1))
v = unlist(error_list_train)
names(v) = to_vec(for(i in names(v)) 
  strsplit(i, '.', fixed = TRUE)[[1]][1])
plot(as.numeric(names(v)), v, xaxt="n", col='blue', main = 'Train Error',
     xlab = 'Values of K', ylab = 'Error')
axis(1, at = seq(1, 61, by = 2), las=2)

v = unlist(error_list_test)
names(v) = to_vec(for(i in names(v)) 
  strsplit(i, '.', fixed = TRUE)[[1]][1])
plot(as.numeric(names(v)), v, xaxt="n", col='red', main = 'Test Error',
     xlab = 'Values of K', ylab = 'Error')
axis(1, at = seq(1, 61, by = 2), las=2)

#--------------------- PCA ----------------------
new_train = train_
new_test = test_

new_train$Species = as.numeric(new_train$Species)
new_test$Species = as.numeric(new_test$Species)

pca_train = pcr(Species ~., data = new_train, scale = TRUE, validation = 'none')
summary(pca_train)

pca_test = pcr(Species ~., data = new_test, scale = TRUE, validation = 'none')
summary(pca_test)

new_train = pca_train$scores
new_test = pca_test$scores

new_train = as.matrix(cbind(new_train[ , c("Comp 1", "Comp 2")],
                            as.factor(train_$Species)))
new_test = as.matrix(cbind(new_test[ , c("Comp 1", "Comp 2")],
                           as.factor(test_$Species)))

colnames(new_train) = c('pc_1', 'pc_2', 'Species')
colnames(new_test) = c('pc_1', 'pc_2', 'Species')

#---------------------- K-Nearest Neighbor ------------------------
error_list_train = list() # List created to store Training Error for each value of K
error_list_test = list() # List created to store Testing Error for each value of K

require(class)
for (k in seq(1, 61, 2))
{
  KNN_train = knn(new_train[, -c(3)], new_train[, -c(3)],
                  as.factor(new_train[, c('Species')]), k) # Train Prediction
  KNN_test = knn(new_train[, -c(3)], new_test[, -c(3)],
                 as.factor(new_train[, c('Species')]), k) # Test Prediction
  
  train_predicted = as.factor(KNN_train)
  test_predicted = as.factor(KNN_test)
  
  train_actual = as.factor(new_train[, c('Species')])
  test_actual = as.factor(new_test[, c('Species')])
  
  tab_train = table(train_predicted, train_actual)
  tab_test = table(test_predicted, test_actual)
  
  #-------- Confusion Matrix for Accuracy ---------
  conf_train = confusionMatrix(tab_train)
  conf_test = confusionMatrix(tab_test)
  
  error_list_train[[glue('{k}')]] = 1 - round(
    conf_train$overall['Accuracy'], 4)
  error_list_test[[glue('{k}')]] = 1 - round(
    conf_test$overall['Accuracy'], 4)
  
}

#------------ Plotting Errors for all values of K ---------------
par(mfrow = c(2,1))
v = unlist(error_list_train)
names(v) = to_vec(for(i in names(v)) 
  strsplit(i, '.', fixed = TRUE)[[1]][1])
plot(as.numeric(names(v)), v, xaxt="n", col='blue', main = 'Train Error',
     xlab = 'Values of K', ylab = 'Error')
axis(1, at = seq(1, 61, by = 2), las=2)

v = unlist(error_list_test)
names(v) = to_vec(for(i in names(v)) 
  strsplit(i, '.', fixed = TRUE)[[1]][1])
plot(as.numeric(names(v)), v, xaxt="n", col='red', main = 'Test Error',
     xlab = 'Values of K', ylab = 'Error')
axis(1, at = seq(1, 61, by = 2), las=2)

# Score Plot on first 2 PC's
pca_res = prcomp(data_[1:4], scale = TRUE)
autoplot(pca_res, data = data_, colour = 'Species')

