# DATA PRE PROCESSING TEMPLATE
# R - mavigobi


# Import dataset
dataset = read.csv('Data.csv')
# Not define  X and y variables
# dataset = dataset[,2:3]

# Overfitting problem. The Countries is  converting to a factor.
# Import packages
# install.packages("caTools")
library(caTools) # import library
set.seed(123) # define seed
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Variables scaling
# It's important remember that Dummy variables are not numbers because Dummy are factors.
# training_set[, 2:3] = scale(training_set[,2:3]) # not : because R assume all rows and columns
# testing_set[,2:3] = scale(testing_set[,2:3])