# Created by Talal Khodr

# Importing Libraries

library(magrittr)
library(lubridate)		  
library(forcats)        
library(dplyr)			    
library(caret)			    
library(ggplot2)		   

library(randomForest)    
library(nnet)            
library(C50)			       
        
library(data.table)
library(tidyverse)
library(keras)
library(corrplot)
library(caret)
library(e1071)

library(NeuralNetTools)
library(padr)  
library(ggthemes)

setwd("/Users/tkhodr/Desktop/Predictive analytics Projet")
set.seed(1995)
credit <- read_csv("/Users/tkhodr/Desktop/Predictive analytics Projet/UCI_Credit_Card.csv", header = TRUE)

options(scipen=999) # So numbers dont show as an exponents

# Data Exploration 

dim(credit)

names(credit)

str(credit)

count(is.null(credit)) # 0

sum(is.na(credit)) # 0

head(credit)
tail(credit)

mean(credit$LIMIT_BAL)
#167484.3
max(credit$LIMIT_BAL)
min(credit$LIMIT_BAL)

mean(credit$EDUCATION)
# 1.853133 This is a high eduation score as a mean

unique(credit$MARRIAGE)

# Data Manipulation

# Factorizing everything
# Sex
credit$SEX <- as.factor(credit$SEX)

levels(credit$SEX) <- c("Male","Female")


#Martial Status
credit$MARRIAGE <- as.factor(credit$MARRIAGE)

levels(credit$MARRIAGE) <- c("Unknown" , "Married" , "Single" ,"Others")

                             
                            
#Education
credit$EDUCATION <- as.factor(credit$EDUCATION)

levels(credit$EDUCATION) <- c(
  "PhD",
  "Graduate school",
  "University",
  "High school",
  "Others",
  "Min Edu",
  "No Edu"
)

#Changing the last variable
credit$default.payment.next.month <- as.factor(credit$default.payment.next.month)

levels(credit$default.payment.next.month) <- c("NO" , "YES") # from binary to str

#Change all the columns into factors
credit$PAY_0 <-as.factor(credit$PAY_0)
credit$PAY_2 <- as.factor(credit$PAY_2)
credit$PAY_3 <- as.factor(credit$PAY_3)
credit$PAY_4 <- as.factor(credit$PAY_4)
credit$PAY_5 <- as.factor(credit$PAY_5)
credit$PAY_6 <- as.factor(credit$PAY_6)

str(credit)

# Exploraorty Data Analysis

# Men vs Female
plot(credit$SEX)

# Age versus Education
ggplot(data = credit, aes(x = AGE , fill= EDUCATION )) +
  geom_histogram( bins = 30, alpha = 0.6 ) # 30 bins is the sweet spot ( Trial and Error )

# Age versus marriage
ggplot(data = credit, aes(x = AGE , fill= MARRIAGE )) +
  geom_histogram( bins = 30, alpha = 0.6 )


# Default number of payments
ggplot(credit, aes(x = AGE, fill = default.payment.next.month)) +
  geom_bar() +
  labs(x = 'Age')

# Default vs marriage
ggplot(credit, aes(x = MARRIAGE, fill = default.payment.next.month)) +
  geom_bar() +
  labs(x = 'Martial Status')

# Default of sexes
ggplot(credit, aes(x = SEX, fill = default.payment.next.month)) +
  geom_bar() +
  labs(x = 'Sex')

#Default on Education
ggplot(credit, aes(x = EDUCATION, fill = default.payment.next.month)) +
  geom_bar() +
  labs(x = 'Education')

#EDA Super Tool
create_report(credit)

# MODELING
# Training and Testing split
credit_no_id <- credit[-c(1)]

set.seed(1995)
credit_m <- as.data.frame(credit_no_id)
credit_sampling_vector <- createDataPartition(credit_m$AGE, p = 0.70, list = FALSE)
credit_train <- credit_m[credit_sampling_vector,]
credit_test <-  credit_m[-credit_sampling_vector,]

# Linear model for fun run on LIMIT_BAL

linear_m <- lm(LIMIT_BAL~., credit_train)
summary(linear_m)
plot(linear_m)

# Tree time 

rf_credit <- randomForest(default.payment.next.month~., data = credit_train, ntree= 500, mtry = 2, importance=TRUE)


prediction_credit <- predict(rf_credit, newdata=credit_test, type="class")


misclassification_error_rate <- sum(credit_test$default.payment.next.month != prediction_credit) / nrow(credit_test)*100
misclassification_error_rate 

# 18.11514

sum(is.na(credit_train))

train.tree <- rpart(default.payment.next.month~., data = credit_train,method = "class")

summary(train.tree)

plot(train.tree)
text(train.tree)

fancyRpartPlot(model = train.tree, main = "Tree")

printcp(train.tree)
plotcp(train.tree)


dtree1 <-
  rpart(
    formula = default.payment.next.month ~ . ,
    data = credit_train,
    method = "class",
    parms = list(split = "information"),
    control = rpart.control(
      cp = 0.01,
      maxcompete = 3,
      minbucket = 5,
      maxsurrogate = 3,
      xval = 20,
      maxdepth = 4
    )
  )
rpart.plot(dtree1)

dtree2 <-
  
  rpart::rpart(
    formula = SEX ~ . ,
    data = credit_train,
    method = "class",
    parms = list(split = "information"),
  )
# printcp(decision_tree_model_one)

rpart.plot(dtree2)

options(scipen=999) # So numbers dont show as an exponents
dtree3 <-
  rpart(
    formula = EDUCATION ~ . ,
    data = credit_train,
    method = "class",
    parms = list(split = "information"),
  )

rpart.plot(dtree3)

dtree4 <-
  rpart(
  formula = SEX,AGE,EDUCATION,default.payment.next.month,MARRIAGE ~.,
    data = credit_train,
    method = "class",
    parms = list(split = "information"),
  )
  
rpart.plot(dtree4)


library(randomForest)

set.seed(1995)

rf.train =randomForest(PAY_0~SEX+EDUCATION+default.payment.next.month,data=credit_train,
                          importance=TRUE)
rf.train

importance(rf.train)
plot(rf.train)
randomForest::varImpPlot(rf.train)
