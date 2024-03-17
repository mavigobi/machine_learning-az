# DATA PRE PROCESSING TEMPLATE: MISSING DATA
# R - mavigobi


# Import dataset
dataset = read.csv('Data.csv')
# Not define  X and y variables


# Nas data processing
# Ifelse is used to verify a condition
# If TRUE, the first part of code is executed
# If FALSE, the second part of code is executed
dataset$Age = ifelse(is.na(dataset$Age), # is.condition? The solution can be true or false
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), # ave produces a subset of mean x values from observations. FUN is the function principal of ave.
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), 
                        dataset$Salary)