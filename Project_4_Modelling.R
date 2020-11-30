setwd("C:/Users/Akash/Desktop/Study/PGP BABI/Course5_Predictive Modelling/Project 4")
library(dplyr)
library(corrplot)
library(ggplot2)
library(tidyverse)
library(lattice)
library(caret)
library(MASS)
library(cluster)
library(randomForest)
library(party)
library(readxl)
library(mice)
library(tidyverse)
library(EnvStats)
library(psych)
library(caTools)
library(car)
library(data.table)
library(scales)
library(ROCR)
library(ineq)
library(InformationValue)
library(class)
library(e1071)
basedataset=read_xlsx("Cellphone.xlsx", sheet = "Data")
dataset=basedataset
names(dataset)
str(dataset)
dataset$Churn=as.factor(dataset$Churn)
dataset$ContractRenewal=as.factor(dataset$ContractRenewal)
dataset$DataPlan=as.factor(dataset$DataPlan)
str(dataset)
summary(dataset)
#Variable description	
#Churn -	1 if customer cancelled service, 0 if not (dependent var, categorical var)
#AccountWeeks -	number of weeks customer has had active account
#ContractRenewal -	1 if customer recently renewed contract, 0 if not (categorical var)
#DataPlan -	1 if customer has data plan, 0 if not (categorical var)
#DataUsage -	gigabytes of monthly data usage
#CustServCalls -	number of calls into customer service
#DayMins -	average daytime minutes per month
#DayCalls -	average number of daytime calls
#MonthlyCharge -	average monthly bill
#OverageFee -	largest overage fee in last 12 months
#RoamMins -	average number of roaming minutes

# (1) Exploratory Data Analysis - Univariate & bivariate analysis, Outliers &
#                                 missing value treatment, multicollinearity treatment

# a) Univariate and bivariate analysis
attach(dataset)

# univariate analysis
hist(AccountWeeks, main='No. of Weeks with Active Account')
hist(DataUsage, main='Monthly data usage in gb') # highest frequecy of 0-0.5 gb/month data usage
hist(CustServCalls, main='No. of calls to Cust. Service') # highest for 1 customer care call 
hist(DayMins, main='Avg. day time min. per month')
hist(DayCalls, main='Avg. day time calls') # left skewed curve
boxplot(DayCalls) # One of the customer seems to had very low day calls, might be 
                  # customer service or network issue. Needs RCA
hist(MonthlyCharge, main='Avg. monthly bill') # right skewed curve
boxplot(MonthlyCharge) # no abnomalies. Some customers have taken higher plan
hist(OverageFee, main='Largest overage fee in LTM')
hist(RoamMins, main='Avg. no. of roaming min.')
barplot(table(Churn), main='Freq. of Churns', xlab='Churn Category', ylab='Count')
table(Churn) # frequency of 0 and 1 in churn
barplot(table(ContractRenewal),main='Freq. of Renewal', xlab='Churn Category', ylab='Count')
table(ContractRenewal) # frequency of 0 and 1 in contract renewal
barplot(table(DataPlan),main='Freq. of Data Plan', xlab='Churn Category', ylab='Count')
table(DataPlan) # frequency of 0 and 1 in having a data plan

# churn % i.e. cancelled service in existing data
Perc_Churn=(table(Churn)[2]/(table(Churn)[2]+table(Churn)[1]))*100 
Perc_Churn

# bivariate analysis
boxplot(AccountWeeks~Churn, col=c("red","blue")) # not much impact
boxplot(DataUsage~Churn,col=c("red","blue")) # higher data usage slightly lower chance of churn
boxplot(CustServCalls~Churn,col=c("red","blue")) # more cust. serv. call higher chance of churn
boxplot(DayMins~Churn,col=c("red","blue")) # high daymin. higher chance of churn
boxplot(DayCalls~Churn,col=c("red","blue")) # not much impact
boxplot(MonthlyCharge~Churn,col=c("red","blue")) # higher monthly charge higher churn
boxplot(OverageFee~Churn,col=c("red","blue")) # not much impact
boxplot(RoamMins~Churn,col=c("red","blue")) # not much impact

# b) Outliers and missing value treatment
sapply(dataset, function(x) sum(is.na(x))) # no missing values in the dataset

# We have observed no outliers from a business dtand point as
# as none of the vales some to be abnormal in nature

# c) Muilticollinearity Treatment
dataset_mc=basedataset
cor(dataset_mc)
corrplot.mixed(cor(dataset_mc), lower="number",
               tl.col="black",tl.pos="lt",number.cex=0.5, tl.cex=0.5)
vifmatrix=vif(glm(Churn~.,data=dataset,family = binomial(link="logit")))
vifmatrix #test for multicollinearity in independent variable using VIF 
#(variance inflation factor) considering threshold of 3

# treatment for multicollinearity using PCA
components=principal(dataset[,-c(1:4,6,8,10:11)],nfactors=1,rotate="varimax")
components #extracting factors without rotation
fa.diagram(components) #the component analysis plot clearly does not
dataset1=cbind(dataset[,c(1:4,6,8,10:11)],components$scores)
head(dataset1)
colnames(dataset1)[9]=c("MonthlyPlan")
head(dataset1)

# only for corrplot
components=principal(dataset_mc[,-c(1:4,6,8,10:11)],nfactors=1,rotate="varimax")
components #extracting factors without rotation
fa.diagram(components) #the component analysis plot clearly does not
dataset1_mc=cbind(dataset_mc[,c(1:4,6,8,10:11)],components$scores)
head(dataset1_mc)
colnames(dataset1_mc)[9]=c("MonthlyPlan")
head(dataset1_mc)

# check for multicollinearity successful with vif threshold of 3
cor(dataset1_mc)
corrplot.mixed(cor(dataset1_mc), lower="number",
               tl.col="black",tl.pos="lt",number.cex=0.5, tl.cex=0.5)
vifmatrix=vif(glm(dataset1$Churn~.,data=dataset1, family = binomial(link="logit")))
vifmatrix

# (2) Model Building - Logistic Regression, KNN, NB and Model Comparison

seed=1000
set.seed(seed)
dataset2=dataset1
dataset2[,-c(1,3,4)]=scale(dataset2[,-c(1,3,4)])
sample=sample.split(dataset2$Churn,SplitRatio = 0.7)
train=subset(dataset2,sample==T)
test=subset(dataset2,sample==F)

# a) Logistic Regression
train_LR=train
test_LR=test
LR_model=glm(train_LR$Churn~., family=binomial(link="logit"),data=train_LR)
LR_model
summary(LR_model)
anova(LR_model, test="Chisq") # gives us the 3 most important parameters to be 
                              # ContractRenewal, CustServCalls and MonthlyPlan

# model performance on train dataset
train_LR$Churn_Predict=predict(LR_model,data=train_LR,type='response')
boxplot(train_LR$Churn_Predict)
summary(train_LR$Churn_Predict)
train_LR$Churn_Predict=ifelse(train_LR$Churn_Predict<0.5,0,1)
train_LR_tbl=table(train_LR$Churn,train_LR$Churn_Predict)
train_LR_tbl # train confusion matrix
train_LR_Accuracy=(train_LR_tbl[1,1]+train_LR_tbl[2,2])/sum(train_LR_tbl)
train_LR_Accuracy # model accuracy
train_predict_churn=train_LR_tbl[2,2]/(train_LR_tbl[2,1]+train_LR_tbl[2,2])
train_predict_churn # ability to predict churn

# odds ratio analysis
exp(cbind(OR=coef(LR_model),confint(LR_model)))

train_LR_predobj <- prediction(train_LR$Churn_Predict, train_LR$Churn)
train_LR_perf <- performance(train_LR_predobj, "tpr", "fpr")
plot(train_LR_perf)
train_LR_KS <- max(attr(train_LR_perf, 'y.values')[[1]]-attr(train_LR_perf, 'x.values')[[1]])
train_LR_auc <- performance(train_LR_predobj,"auc"); 
train_LR_auc <- as.numeric(train_LR_auc@y.values)
train_LR_gini = ineq(train_LR$Churn_Predict, type="Gini")
train_LR_auc
train_LR_KS
train_LR_gini
Concordance(actuals = train_LR$Churn,predictedScores = train_LR$Churn_Predict)

# model performance on test dataset
test_LR$Churn_Predict= predict(LR_model,newdata=test_LR,type='response')
boxplot(test_LR$Churn_Predict)
summary(test_LR$Churn_Predict)
test_LR$Churn_Predict=ifelse(test_LR$Churn_Predict<0.5,0,1)
test_LR_tbl=table(test_LR$Churn,test_LR$Churn_Predict)
test_LR_tbl # test confusion matrix
test_LR_Accuracy=(test_LR_tbl[1,1]+test_LR_tbl[2,2])/sum(test_LR_tbl)
test_LR_Accuracy # model accuracy
test_predict_churn=test_LR_tbl[2,2]/(test_LR_tbl[2,1]+test_LR_tbl[2,2])
test_predict_churn # ability to predict churn

test_LR_predobj <- prediction(test_LR$Churn_Predict, test_LR$Churn)
test_LR_perf <- performance(test_LR_predobj, "tpr", "fpr")
plot(test_LR_perf)
test_LR_KS <- max(attr(test_LR_perf, 'y.values')[[1]]-attr(test_LR_perf, 'x.values')[[1]])
test_LR_auc <- performance(test_LR_predobj,"auc"); 
test_LR_auc <- as.numeric(test_LR_auc@y.values)
test_LR_gini = ineq(test_LR$Churn_Predict, type="Gini")
test_LR_auc
test_LR_KS
test_LR_gini
Concordance(actuals = test_LR$Churn,predictedScores = test_LR$Churn_Predict)

# b) KNN (K-Nearest Neighbours)

train_KNN=train
test_KNN=test
test_KNN$Churn_Predict=knn(train_KNN[,-1], test_KNN[,-1], train_KNN[,1], k = 19) 
test_KNN$Churn_Predict=as.vector(test_KNN$Churn_Predict, mode="numeric")
test_KNN$Churn=as.vector(test_KNN$Churn, mode="numeric")
test_KNN_tbl = table(test_KNN$Churn, test_KNN$Churn_Predict)
test_KNN_tbl # test confusion matrix
test_KNN_Accuracy=sum(diag(test_KNN_tbl)/sum(test_KNN_tbl)) 
test_KNN_Accuracy # model accuracy
test1_predict_churn=test_KNN_tbl[2,1]/(test_KNN_tbl[2,1]+test_KNN_tbl[1,1])
test1_predict_churn # ability to predict churn

test_KNN_predobj <- prediction(test_KNN$Churn_Predict, test_KNN$Churn)
test_KNN_perf <- performance(test_KNN_predobj, "tpr", "fpr")
plot(test_KNN_perf)
test_KNN_KS <- max(attr(test_KNN_perf, 'y.values')[[1]]-attr(test_KNN_perf, 'x.values')[[1]])
test_KNN_auc <- performance(test_KNN_predobj,"auc"); 
test_KNN_auc <- as.numeric(test_KNN_auc@y.values)
test_KNN_gini = ineq(test_KNN$Churn_Predict, type="Gini")
test_KNN_auc
test_KNN_KS
test_KNN_gini
Concordance(actuals = test_KNN$Churn,predictedScores = test_KNN$Churn_Predict)

# c) NB (Naive Bayes)

train_nb=train
test_nb=test

NB_model = naiveBayes(train_nb$Churn~., data = train_nb)
NB_model
summary(NB_model)

# model performance on train dataset
train_nb$Churn_Predict=predict(NB_model, train_nb, type = "class")
summary(train_LR$Churn_Predict)

view(train_nb)

train_nb_tbl=table(train_nb$Churn,train_nb$Churn_Predict)
train_nb_tbl # train confusion matrix
train_nb_Accuracy=(train_nb_tbl[1,1]+train_nb_tbl[2,2])/sum(train_nb_tbl)
train_nb_Accuracy # model accuracy
train3_predict_churn=train_nb_tbl[2,2]/(train_nb_tbl[2,1]+train_nb_tbl[2,2])
train3_predict_churn # ability to predict churn

#train_nb$Churn=as.numeric(train_nb$Churn)
#train_nb$Churn_Predict=as.numeric(train_nb$Churn_Predict)
str(train_nb)
train_nb$Churn_Predict=as.numeric(train_nb$Churn_Predict)
train_nb$Churn=as.numeric(train_nb$Churn)
train_nb_predobj <- prediction(train_nb$Churn_Predict, train_nb$Churn)
train_nb_perf <- performance(train_nb_predobj, "tpr", "fpr")
plot(train_nb_perf)
train_nb_KS <- max(attr(train_nb_perf, 'y.values')[[1]]-attr(train_nb_perf, 'x.values')[[1]])
train_nb_auc <- performance(train_nb_predobj,"auc"); 
train_nb_auc <- as.numeric(train_nb_auc@y.values)
train_nb_gini = ineq(train_nb$Churn_Predict, type="Gini")
train_nb_auc
train_nb_KS
train_nb_gini
Concordance(actuals = train_nb$Churn,predictedScores = train_nb$Churn_Predict)

# model performance on test dataset
test_nb$Churn_Predict=predict(NB_model, test_nb, type = "class")
summary(test_nb$Churn_Predict)

test_nb_tbl=table(test_nb$Churn,test_nb$Churn_Predict)
test_nb_tbl # train confusion matrix
test_nb_Accuracy=(test_nb_tbl[1,1]+test_nb_tbl[2,2])/sum(test_nb_tbl)
test_nb_Accuracy # model accuracy
test3_predict_churn=test_nb_tbl[2,2]/(test_nb_tbl[2,1]+test_nb_tbl[2,2])
test3_predict_churn # ability to predict churn

test_nb$Churn=as.numeric(test_nb$Churn)
test_nb$Churn_Predict=as.numeric(test_nb$Churn_Predict)

test_nb_predobj <- prediction(test_nb$Churn_Predict, test_nb$Churn)
test_nb_perf <- performance(test_nb_predobj, "tpr", "fpr")
plot(test_nb_perf)
test_nb_KS <- max(attr(test_nb_perf, 'y.values')[[1]]-attr(test_nb_perf, 'x.values')[[1]])
test_nb_auc <- performance(test_nb_predobj,"auc"); 
test_nb_auc <- as.numeric(test_nb_auc@y.values)
test_nb_gini = ineq(test_nb$Churn_Predict, type="Gini")
test_nb_auc
test_nb_KS
test_nb_gini
Concordance(actuals = test_nb$Churn,predictedScores = test_nb$Churn_Predict)
