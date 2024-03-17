# DATA PRE PROCESSING TEMPLATE: CATEGORICAL DATA
# R - mavigobi


# Import dataset
dataset = read.csv('Data.csv')
# Not define  X and y variables


# Categorical variables
# Factor amounts to whole a categorical variables
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0, 1)) 