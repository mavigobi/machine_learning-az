# MULTIPLE LINEAR REGREESION
# R - mavigobi


#PRE PROCESSING DATA
# Import dataset
dataset = read.csv('50_Startups.csv')
# Not define  X and y variables


# Categorical variables
# Factor amounts to whole a categorical variables
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1, 2, 3))


# Overfitting problem. The Countries is  converting to a factor.
# Import packages
# install.packages("caTools")
library(caTools) # import library
set.seed(123) # define seed
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Multiple linear regression: training data
#regression = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State) not recommended
regression = lm(Profit ~ .,
                data = training_set)


# Prediction of data
y_pred = predict(regression, newdata = testing_set)


# BACKWARD ELIMATION OF VARIABLES
SL = 0.05
# dataset = dataset[, c(1,2,3,4,5)]
dataset = dataset[, c(1:length(dataset))]

backward_elimination <- function (data_function, sl)
{
  for (j in c(1:length(data_function)))
  {
    regression = lm(formula = Profit ~ ., data = data_function)
    coefs = summary(regression)$coefficients[-1, "Pr(>|t|)"]
    if (max(coefs) > sl)
    {
      position = which(coefs == max(coefs))
      data_function = data_function[-position]
    }
  }
  return(summary(regression))
}

backward_elimination(training_set, SL)
