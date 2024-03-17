# SIMPLE LINEAR REGRESSION
# R - mavigobi


# PRE PROCESSING DATA

# Import dataset
dataset = read.csv('Salary_Data.csv')
# Not define  X and y variables


# install.package("caTools")http://127.0.0.1:38603/graphics/plot_zoom_png?width=747&height=660
library(caTools) # import library
set.seed(123) # define seed
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Simple linear regression dataset: regression variable
regression = lm(formula = Salary ~ YearsExperience, # dependent variable ~ independent variable
               data = training_set) # it's interesting run: summary(regression)


# DATA PREDICTION
y_pred = predict(regression, newdata = testing_set)


# DISPLAY SIMPLE LINEAR REGRESSION: TRAINING DATA 
#install.packages("ggplot2")
library("ggplot2") # it need representation layers
ggplot() + # initialization ggplot2
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             colour = "blue") + # aesthetic: appearance of points 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regression, newdata =  training_set)), 
              colour = "red") +  # aesthetic: appearance of fit
  ggtitle("Salary vs Experience")+
  xlab("Employee experience (years)") +
  ylab("Employee salary ($)")


# DISPLAY SIMPLE LINEAR REGRESSION: TESTING DATA 
ggplot() + # initialization ggplot2
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary), 
             colour = "blue") + # aesthetic: appearance of points 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regression, newdata =  training_set)), 
            colour = "red") +  # aesthetic: appearance of fit
  ggtitle("Salary vs Experience")+
  xlab("Employee experience (years)") +
  ylab("Employee salary ($)")