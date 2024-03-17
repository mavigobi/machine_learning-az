# SUPPORT VECTOR REGRESSION (SVR)
# R - mavigobi


# PRE PROCESSING DATA

# Import dataset
dataset = read.csv('Position_Salaries.csv')
# Not define  X and y variables
dataset = dataset[,2:3]

# Overfitting problem. 
# Import packages
# install.packages("caTools")
# library(caTools) # import library
# set.seed(123) # define seed
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)

# Variables scaling
# It's important remember that Dummy variables are not numbers because Dummy are factors.
# training_set[, 2:3] = scale(training_set[,2:3]) # not : because R assume all rows and columns
# testing_set[,2:3] = scale(testing_set[,2:3])


# SVR REGRESSION MODEL
# should create "regression" own regression mode
#install.packages("e1071")
library("e1071")
regression = svm(formula = Salary ~.,
                 data = dataset,
                 type = "eps-regression",
                 kernel = "radial")

  
# PREDICTION OF REGRESSION MODEL
data_predict = predict(regression, newdata = data.frame(Level = 6.5))


# DISPLAY OF REGRESSION MODEL
library(ggplot2)
# x_grid increase number of data
#x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "blue") +
  geom_line(aes(x = dataset$Level, y = predict(regression, newdata = data.frame(Level = dataset$Level))),
            color = "red")+
  ggtitle("Modelo de Regresión SVR")+
  xlab("Nivel del empleado") +
  ylab("Salario anual del empleado")