# ORIGINAL BACKWARD ELIMINATION VARIABLES
# R - mavigobi


#PRE PROCESSING DATA
# Import dataset
dataset = read.csv('50_Startups.csv')
# Not define  X and y variables


# BACKWARD ELIMATION OF VARIABLES
# install.packages("stringr")
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