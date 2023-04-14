library(glmnet)

# Getting the independent variable
x_var <- data.matrix(mtcars[, c("hp", "wt", "drat")])
# Getting the dependent variable
y_var <- mtcars[, "mpg"]
best_lambda = 80
# Rebuilding the model with optimal lambda value
best_ridge <- glmnet(x_var, y_var, alpha = 0, lambda = best_lambda)
# here x is the test dataset
pred <- predict(best_ridge, s = best_lambda, newx = x_var)

mean(pred)#20.09062
mean(y_var)#20.09062

library(rpart)
regTree =rpart(mpg ~ hp + wt + drat, data = mtcars)
mean(predict(regTree))#20.09062
