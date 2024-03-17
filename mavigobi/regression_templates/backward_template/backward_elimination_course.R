# BACKWARD ELIMINATION VARIABLES COURSE
# R - mavigobi


# NOT OPTIMAL BACKWARD ELIMINATION METHOD
#SL = 0.05

#regression = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
#summary(regression)

#regression = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
#summary(regression)

#regression = lm(Profit ~ R.D.Spend + Marketing.Spend,data = dataset)
#summary(regression)

#regression = lm(Profit ~ R.D.Spend, data = dataset)
#summary(regression)


# COURSE BACKWARD ELIMINATION METHOD 
SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]


# AUTOMATION OF BACKWARD VARIABLE ELIMINATION
backwardElimination <- function(x, sl) # backwardElimination function
{
  numVars = length(x) # numVars save the length of matrix  
  for (i in c(1:numVars)) # iterates through the array from 1 to numVars
  { 
    regressor = lm(formula = Profit ~ ., data = x) 
    maxVar_backup = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"]) # gets the max p-value
    R2_backup = summary(regressor)$adj.r.squared # save the result R^2 before deleting that column
    if (maxVar_backup > sl)
    { # comparative between maxVar and SL
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar_backup) #takes the column index and stores it in j
      x_backup = x[,-j] # create a copy of the matrix x without that column (variable) that has the maximum p-value 
      temp_regressor = lm(formula = Profit ~., data =x_backup) # new regression of x without j column
      maxVar_now = max(coef(summary(temp_regressor))[c(2:numVars-1), "Pr(>|t|)"]) # gets the max p-value
      R2_now = summary(temp_regressor)$adj.r.squared
    }
    if (maxVar_now <= maxVar_backup) # it's interesting to make a change of R^2
    {
      if (R2_backup < R2_now)
      {
        x = x_backup # make a change of x
        numVars = numVars - 1 # numVars has a less column
      }
      
    }
  }
  regressor = lm(formula = Profit ~ ., data = x) 
  return(summary(regressor))
}

backwardElimination(training_set,SL)

