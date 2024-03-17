# DECISION TREE REGREESION MODEL
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


# DECISION TREE REGRESSION MODEL
# should create "regression" own regression mode
#install.packages("rpart")
library("rpart")
regression = rpart(formula = Salary ~.,
                   data = dataset,
                   control = rpart.control(minsplit = 1))

# PREDICTION OF REGRESSION MODEL
data_predict = predict(regression, newdata = data.frame(Level = 6.5))


# DISPLAY OF  REGRESSION MODEL
library(ggplot2)
# x_grid increase number of data
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "blue") +
  geom_line(aes(x = x_grid, y = predict(regression, newdata = data.frame(Level = x_grid))),
            color = "red")+
  ggtitle("Modelo de Regresión: predicción con árbol de decisión ")+
  xlab("Nivel del empleado") +
  ylab("Salario anual del empleado")


