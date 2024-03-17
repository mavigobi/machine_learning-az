# POLYNOMIAL REGREESION MODEL
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


# LINEAR REGRESSION MODEL

l_regression = lm(formula = Salary ~ ., 
                  data = dataset)
# summary(l_regression)


# POLYNOMIAL REGRESSION MODEL
# add as many columns as degrees of the polynomial
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_regression = lm(formula = Salary ~ .,
                     data = dataset)
# summary(poly_regression)


# DISPLAY OF LINEAR REGRESSION MODEL
library("ggplot2")
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "blue") +
  geom_line(aes(x = dataset$Level, y = predict(l_regression, newdata = dataset)),
            color = "red") +
  ggtitle("Modelo de Regresión Lineal") +
  xlab("Nivel del empleado")
  ylab("Salario anual del empleado")
  
# DISPLAY OF POLYNOMIAL REGRESSION MODEL
  ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
               color = "blue")+
    geom_line(aes(x = dataset$Level, y = predict(poly_regression, newdata = dataset)),
              color = "red")+
    ggtitle("Modelo de Regresión Polinómica")+
    xlab("Nivel del empleado") +
    ylab("Salario anual del empleado")
  
  
  
# DISPLAY OF POLYNOMIAL REGRESSION MODEL (HIGH ACCURACY)
library(ggplot2)
# x_grid increase number of data
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
            color = "blue")+
  geom_line(aes(x = x_grid, y = predict(poly_regression, newdata = data.frame(Level = x_grid,
                                                                                     Level2 = x_grid^2,
                                                                                     Level3 = x_grid^3,
                                                                                     Level4 = x_grid^4))),
            color = "red")+
  ggtitle("Modelo de Regresión Polinómica")+
  xlab("Nivel del empleado") +
  ylab("Salario anual del empleado")
  
  
# PREDICTION OF LINEAR REGRESSION MODEL
y_predict = predict(l_regression, newdata = data.frame(Level = 6.5))
  
# PREDICTION OF POLYNOMIAL REGRESSION MODEL
y_poly_predict = predict(poly_regression, newdata = data.frame(Level = 6.5,
                                                          Level2 = 6.5^2,
                                                          Level3 = 6.5^3,
                                                          Level4 = 6.5^4))


