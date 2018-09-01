library(boot)
library(car)
library(QuantPsyc)
library(sandwich)
library(vars)
library(nortest)
library(MASS)

house <- read.csv("C:\\Users\\lenovo\\Documents\\R\\Data.csv")
summary(house)

sapply(house, function(x) sum(is.na(x)))
house <- na.omit(house)


boxplot(house$Taxi_dist)

quantile(house2$Price_house, c(0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
boxplot(house2$Price_house)
house2 <- house[house$Price_house < 9948270, ]

quantile(house$Taxi_dist, c(0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
house2 <- house[house$Taxi_dist >146, ]
boxplot(house2$Taxi_dist)

quantile(house$Market_dist, c(0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
house2 <- house[house$Market_dist > 1666,]
boxplot(house2$Market_dist)

quantile(house$Hospital_dist, c(0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
boxplot(house2$Hospital_dist)
house2 <- house[house$Hospital_dist > 3227, ]
house2 <- house[house$Hospital_dist < 188421, ]

quantile(house$Carpet_area, c(0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
boxplot(house2$Carpet_area)
house2 <- house[house$Carpet_area < 2027, ]

quantile(house$Builtup_area, c(0,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
boxplot(house2$Builtup_area)
house2 <- house[house$Builtup_area < 2421, ]

fit <- lm(Price_house ~ Taxi_dist + Market_dist + Hospital_dist+ Carpet_area + Builtup_area + Parking_type + City_type + Rainfall, data=house2)
summary(fit)


fit <- lm(Price_house ~ Taxi_dist + Market_dist + Hospital_dist+ Builtup_area + Parking_type + City_type + Rainfall, data=house2)
fit <- lm(Price_house ~ Taxi_dist  + Hospital_dist+ Builtup_area + Parking_type + City_type + Rainfall, data=house2)
fit <- lm(Price_house ~ Hospital_dist+ Builtup_area + Parking_type + City_type + Rainfall, data=house2)
fit <- lm(Price_house ~ Hospital_dist+ Builtup_area + Parking_type + City_type , data=house2)

attach(house2)
vif(fit)

resids <- fit$residuals

percent_diff <- resids/house2$Price_house
percent_diff
abs(percent_diff)

mean(abs(percent_diff))

fitted(fit)
house2$pred <- fitted(fit)

#>0.05 = Homoscedascity
bptest(fit)
dwtest(fit)

ad.test(resids)








