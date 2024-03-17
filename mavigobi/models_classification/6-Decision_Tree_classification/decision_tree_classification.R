# CLASSIFICATION OF DECISION TREE
# R - mavigobi


# PRE PROCESSING DATA

# Import dataset
dataset = read.csv('Social_Network_Ads.csv')
# Not define  X and y variables
dataset = dataset[,3:5]


# CODE THE VARIABLES RANKING AS A FACTOR
dataset$Purchased = factor(dataset$Purchased,
                          levels = c(0,1))

# Overfitting problem. The Countries is  converting to a factor.
# Import packages
# install.packages("caTools")
library(caTools) # import library
set.seed(123) # define seed
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# NOT NECESSARY BECAUSE THE CLASSIFIER OF DECISION TREE MEASURES THE MINIMUM ENTROPY

# Variables scaling
# It's important remember that Dummy variables are not numbers because Dummy are factors.
#training_set[, 1:2] = scale(training_set[,1:2]) # not : because R assume all rows and columns
#testing_set[,1:2] = scale(testing_set[,1:2])


# CLASSIFIER OF LOGISTIC MODEL REGRESSION 
library(rpart)
classifier = rpart(formula = Purchased~.,
                   data = training_set)

# PREDICTION OF OWN CLASSIFIER LOGISTIC MODEL REGRESSION 
y_pred = predict(classifier, 
                 newdata = testing_set[,-3],
                 type = "class")

# CREATE A CONFUSION MATRIX
cm = table(testing_set[,3], y_pred)


# VISUALITATION OF DATA: TARINING DATA
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 500)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree Classification (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))

# VISUALITATION OF DATA: TESTING DATA
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 500)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3], main = 'Decision Tree Classification (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))

# Plotting the tree
plot(classifier)
text(classifier)