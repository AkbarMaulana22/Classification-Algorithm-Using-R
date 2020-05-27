#Required Packages
library(xlsx)
library(kernlab)
library(caret)
library(e1071)        
library(ISLR)         
library(RColorBrewer) 
library(ggplot2)
library(rpart)
library(rattle)
library(class)
library(randomForest)
library(randomForestSRC)
library(ggplot2)
library(pca3d)
library(gridExtra)
library(car)
library(MASS)
library(fpc)
library(ROSE)
library(party)
library(kknn)
library(RWeka)
library(corrplot)
library(rpart.plot)
library(psych)
library(mice)

#Load Dataset
data=read.xlsx(file.choose(),sheetName="Sheet1")

#Statistik Deskriptif
summary(data)

#Preprocessing
#Deteksi Missing value
data[which(data=="NA"),]
which(data=="NA")
is.na(data)
#Mean imputation
for(i in 1:ncol(data)) {
  data[ , i][is.na(data[ , i])] <- mean(data[ , i], na.rm = TRUE)
}
#Median imputation
for(i in 1:ncol(data)) {
  data[ , i][is.na(data[ , i])] <- median(data[ , i], na.rm = TRUE)
}

#Deteksi Outlier
win.graph()
par(mfrow=c(1,3))
for(i in 1:3) {
  boxplot(data[,i], main=names(data)[i],col="red")
}
attach(data)
featurePlot(x=data[,-4], y=data[,4], plot="box")

#EDA
featurePlot(x=data[,-4], y=data[,4], plot="ellipse")

#Feature extraction
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=data[,-4], y=data[,4], plot="density", scales=scales)

box1 <- ggplot(data=data, aes(x=Jenis.Sel, y=Diameter.WBC.µm.)) +
  geom_boxplot(aes(fill=Jenis.Sel)) + 
  ylab("Diameter WBC") +
  ggtitle("Boxplot") +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) 
print(box1)

box2 <- ggplot(data=data, aes(x=Jenis.Sel, y=Rasio.Nukleus)) +
  geom_boxplot(aes(fill=Jenis.Sel)) + 
  ylab("Rasio Nukleus") +
  ggtitle("Boxplot") +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) 
print(box2)

box3 <- ggplot(data=data, aes(x=Jenis.Sel, y=Kebundaran.Nukleus)) +
  geom_boxplot(aes(fill=Jenis.Sel)) + 
  ylab("Kebundaran Nukleus") +
  ggtitle("Boxplot") +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) 
print(box3)

#Korelasi
pairs.panels(data[,-4],gap=0)
cor(data[,-4])
corrplot(cor(data[,-4]),method = "ellipse",type="upper")

qplot(x=Diameter.WBC.µm.,y=Rasio.Nukleus,data=data,col=Jenis.Sel)

#Dimensional Reduction
fit_pca <- princomp(data[,-4], cor = TRUE, scores = TRUE, covmat = NULL)
summary(fit_pca)
loadings(fit_pca)
plot(fit_pca,type="lines")
data_baru=fit_pca$scores
pca2d(fit.pca, group = data[,4])
pca3d(fit.pca, group = data[,4])

#Partisi Data
#Hold Out
train.flag <- createDataPartition(y=data$Jenis.Sel, p=0.8, list=FALSE)
training   <- data[ train.flag, ]
testing <- data[-train.flag, ]

#1.CART
fit.CART=rpart(Jenis.Sel~.,data=training,method="class")
plotcp(fit.CART)
summary(fit.CART)
win.graph()
rpart.plot(fit.CART)

prune2=prune.rpart(fit.CART,cp=fit.CART$cptable[which.min(fit.CART$cptable[,"xerror"]),"CP"])
rpart.plot(prune2)
Summary(prune2)
plotcp(prune2)

prediksi.training=predict(fit.CART,training[,-4],type="class")
prediksi.testing=predict(fit.CART,testing[,-4],type="class")
confusionMatrix(prediksi.training,training$Jenis.Sel)
confusionMatrix(prediksi.testing,testing$Jenis.Sel)


# 2. SVMs
#-----------------------------------------------------------------------------
model.svm <- ksvm(Jenis.Sel~., data=training)
train.svm  <- predict(model.svm, newdata=training, type="response")
confusionMatrix(train.svm,training$Jenis.Sel)
pred.svm <- predict(model.svm, newdata=testing)
confusionMatrix(pred.svm,testing$Jenis.Sel)

#Tune out
tune.out <- tune(svm,train.x=training[,-4],train.y=training[,4],data = training, kernel = "radial",
                 ranges = list(cost = c(0.1,1,10,100,1000),
                               gamma = c(0.5,1,2,3,4)))
tune.out2 <- tune(svm, train.x=training[,-4],train.y=training[,4], data = training, kernel = "linear",
                  ranges = list(cost = c(0.1,1,10,100,1000)))
tune.out3 <- tune(svm, train.x=training[,-4],train.y=training[,4], data = training, kernel = "sigmoid",
                  ranges = list(cost = c(0.1,1,10,100,1000),
                                gamma = c(0.5,1,2,3,4)))

best1=tune.out$best.model
best2=tune.out2$best.model
best3=tune.out3$best.model
pred.svm1 <- predict(best1,testing[,-4])
pred.svm1 <- predict(best1,training[,-4])
confusionMatrix(pred.svm1,testing$Jenis.Sel)
confusionMatrix(pred.svm1,training$Jenis.Sel)
pred.svm2 <- predict(best2, newdata=testing[,-4])
pred.svm2 <- predict(best2, newdata=training[,-4])
confusionMatrix(pred.svm2,testing$Jenis.Sel)
confusionMatrix(pred.svm2,training$Jenis.Sel)
pred.svm3 <- predict(best3, newdata=testing[,-4])
pred.svm3 <- predict(best3, newdata=training[,-4])
confusionMatrix(pred.svm3,testing$Jenis.Sel)
confusionMatrix(pred.svm3,training$Jenis.Sel)

# 3. RANDOM FOREST
#-----------------------------------------------------------------------------
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree", "nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree", "nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, nodesize=param$nodesize, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

#k-fold CV
fitControl <- trainControl(
  method = "repeatedcv",
  number = 3, 
  ## repeated ten times
  repeats = 10)

#Tune.out
grid <- expand.grid(.mtry=c(floor(sqrt(ncol(training))), (ncol(training) - 1), floor(log(ncol(training)))), 
                    .ntree = c(100, 300, 500, 1000),
                    .nodesize =c(1:4))

set.seed(123)
fit_rf <- train(Jenis.Sel ~ ., 
                data = training, 
                method = customRF, 
                tuneGrid= grid,
                trControl = fitControl)

fit_rf$finalModel
win.graph()
plot(fit_rf)
varImportance <- varImp(fit_rf, scale = FALSE)

varImportanceScores <- data.frame(varImportance$importance)

varImportanceScores <- data.frame(names = row.names(varImportanceScores), var_imp_scores = varImportanceScores$B)

plot(varImportanceScores)


ggplot(varImportanceScores, 
       aes(reorder(names, var_imp_scores), var_imp_scores)) + 
  geom_bar(stat='identity', 
           fill = '#875FDB') + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  coord_flip() + 
  labs(x = 'Feature', y = 'Importance') + 
  ggtitle('Feature Importance for Random Forest Model')

plot(varImportance)

oob_error <- data.frame(mtry = seq(1:100), oob = fit_rf$finalModel$err.rate[, 'OOB'])

paste0('Out of Bag Error Rate for model is: ', round(oob_error[100, 2], 4))

ggplot(oob_error, aes(mtry, oob)) +  
  geom_line(colour = 'red') + 
  theme_minimal() + 
  ggtitle('OOB Error Rate across 100 trees') + 
  labs(y = 'OOB Error Rate')

predict_values <- predict(fit_rf, newdata = training[,-4])
confusionMatrix(predict_values,training$Jenis.Sel)
predict_values2 <- predict(fit_rf, newdata = testing[,-4])
confusionMatrix(predict_values2,testing$Jenis.Sel)

#SMOTE RF
fitControl$sampling <- "smote"

smote_fit <- train(Jenis.Sel ~ .,
                   data = training,
                   method = customRF, 
                   tuneGrid= grid,
                   trControl = fitControl)
predict_values1 <- predict(smote_fit, newdata = testing)
confusionMatrix(predict_values,testing$Jenis.Sel)
predict_values2 <- predict(smote_fit, newdata = training)
confusionMatrix(predict_values2,training$Jenis.Sel)

#4. Naive Bayes
model.bayes=naiveBayes(Jenis.Sel ~ .,data=training)
summary(model.bayes)
prediksi.bayes1=predict(model.bayes,training[,-4])
confusionMatrix(prediksi.bayes1,training[,4])
prediksi.bayes2=predict(model.bayes,testing[,-4])
confusionMatrix(prediksi.bayes2,testing[,4])

#5.Under sampling
data.under= ovun.sample(Jenis.Sel ~ ., data = data, method = "under", N = 40,
                            seed = 1)$data
train.flag <- createDataPartition(y=data.under$Jenis.Sel, p=0.8, list=FALSE)
training.under   <- data.under[ train.flag, ]
testing.under <- data.under[-train.flag, ]

#6. Over sampling
data.over= ovun.sample(Jenis.Sel ~ ., data = data, method = "over", N = 1000,
                        seed = 1)$data
train.flag <- createDataPartition(y=data.over$Jenis.Sel, p=0.8, list=FALSE)
training.over   <- data.over[ train.flag, ]
testing.over <- data.over[-train.flag, ]

#7. Both (Under sampling + Over sampling)
data.both= ovun.sample(Jenis.Sel ~ ., data = data, method = "both",p=0.5, N = 1000,
                       seed = 1)$data
train.flag <- createDataPartition(y=data.both$Jenis.Sel, p=0.8, list=FALSE)
training.both   <- data.both[ train.flag, ]
testing.both <- data.both[-train.flag, ]
