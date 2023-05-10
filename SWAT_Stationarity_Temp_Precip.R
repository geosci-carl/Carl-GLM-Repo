# Initial repo population.
rm(list = ls())

library(ggplot2)

## grab PRECIP data ##########
precip <- read.csv("SWAT Input Master Data Precipitation.csv") ## load in data

## grab TEMP data ##########
tempmin <- read.csv("SWAT Input Master Data Temperature - Min.csv") ## load in data
tempmax <- read.csv("SWAT Input Master Data Temperature - Max.csv") ## load in data

# Configure:
loc <- 14 # pick a location from 1 to n

# Let's manage the data
alldata <-as.data.frame(matrix(nrow=nrow(precip),ncol=3)) # daily
colnames(alldata) <- c("precip","tempmin","tempmax")

# select data
# 14,"port14pcp",41.4375,-83.3125,192.4
# Woodville Township, OH
alldata$precip <- precip[,loc]
alldata$tempmax <- tempmax[,loc]
alldata$tempmin <- tempmin[,loc]

# Visualize Halves
length <- nrow(precip)
half <- ceiling(length/2)

data_halves <- as.data.frame(matrix(nrow=nrow(precip),ncol=4))
colnames(data_halves) <- c("half", "precip","tempmin","tempmax")
data_halves$precip <- alldata$precip
data_halves$tempmin <- alldata$tempmin
data_halves$tempmax <- alldata$tempmax
data_halves$half[1:half] <- "1980-2000"
data_halves$half[half:length] <- "2000-2020"

# Plot precip
ggplot(data_halves, aes(x=precip, fill=half))+
  geom_density(alpha=0.4)+
  labs(x="Precipitation [mm]", y="Probability Density",
       title = "PDF for Precip, Woodville, OH, 1980-2020")

# Plot tempmin
ggplot(data_halves, aes(x=tempmin, fill=half))+
  geom_density(alpha=0.4)+
  labs(x="Temperature Min [C]", y="Probability Density",
       title = "PDF for Temperature Minimum, Woodville, OH, 1980-2020")

# Plot tempmax
ggplot(data_halves, aes(x=tempmax, fill=half))+
  geom_density(alpha=0.4)+
  labs(x="Temperature Max [C]", y="Probability Density",
       title = "PDF for Temperature Maximum, Woodville, OH, 1980-2020")




