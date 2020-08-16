#Script Containing the code used for data analysis and production

library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(caret)
library(Hmisc)
library(MASS)
library(randomForest)
housing_data <- read.csv("https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.csv")

head(housing_data)

summary(housing_data)

housing_data$ocean_proximity <- as.factor(housing_data$ocean_proximity)

housing_data$ocean_proximity <- unclass(housing_data$ocean_proximity)

housing_data$ocean_proximity<- as.factor(housing_data$ocean_proximity)

melt.data <- melt(housing_data)

ggplot(data = melt.data, aes(x = value)) + 
  stat_density() + 
  facet_wrap(~ variable, scales = "free")

print("Correlation of Median House Value and Median Income")
cor.test(housing_data$median_house_value, housing_data$median_income)[4]
print("Correlation of Median House Value and Longitude")
cor.test(housing_data$median_house_value, housing_data$longitude)[4]
print("Correlation of Median House Value and Latitude")
cor.test(housing_data$median_house_value, housing_data$latitude)[4]
print("Correlation of Median House Value and Median Age")
cor.test(housing_data$median_house_value, housing_data$housing_median_age)[4]
print("Correlation of Median House Value and Total Rooms")
cor.test(housing_data$median_house_value, housing_data$total_rooms)[4]
print("Correlation of Median House Value and Total BedRooms")
cor.test(housing_data$median_house_value, housing_data$total_bedrooms)[4]
print("Correlation of Median House Value and Population")
cor.test(housing_data$median_house_value, housing_data$population)[4]
print("Correlation of Median House Value and House Holds")
cor.test(housing_data$median_house_value, housing_data$households)[4]

print("Correlation of bedrooms/total_rooms and Median House Value")
cor.test((housing_data$total_bedrooms/housing_data$total_rooms), housing_data$median_house_value)[4]
housing_data$bed_to_room_ratio <- housing_data$total_bedrooms/housing_data$total_rooms

a <- ggplot(data = housing_data, aes(y = median_house_value, x = median_income)) + geom_point()

b <- ggplot(data = housing_data, aes(y = median_house_value, x = housing_median_age)) + geom_point()

c <- ggplot(data = housing_data, aes(y = median_house_value, x = bed_to_room_ratio)) + geom_point()

d <- ggplot(data = housing_data, aes(y = median_house_value, x = total_rooms)) + geom_point()

grid.arrange(a,b,c,d, nrow = 2, ncol = 2)

ggplot() + geom_point(data =  housing_data, aes(x = longitude, y = latitude), color = "green", alpha = 0.2)

ggplot(data = housing_data, aes(y = median_house_value, x = ocean_proximity)) + geom_boxplot()

#MODEL BUILDING 
subset_data <- subset(housing_data, select = c(latitude, longitude, median_income, housing_median_age, bed_to_room_ratio, ocean_proximity, median_house_value))

head(subset_data)

#CODE AFTER THIS IS AVAILABLE IN THE MD AND RMD FILES.
