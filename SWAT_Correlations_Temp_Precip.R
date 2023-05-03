# Initial repo population.
rm(list = ls())

## grab PRECIP data ##########
precip <- read.csv("SWAT Input Master Data Precipitation.csv") ## load in data

## grab TEMP data ##########
tempmin <- read.csv("SWAT Input Master Data Temperature - Min.csv") ## load in data
tempmax <- read.csv("SWAT Input Master Data Temperature - Max.csv") ## load in data

# Data: 14975 observations at 45 locations for 3 different attributes
# precipiation, daily temperature minimum, daily temperature maximum

n = ncol(precip) # iterate over 45 locations

sequence = seq(from=1,to=n,by=1)

#### NON-SPATIAL #####################################################
# Does precip covary with temperature min and max?
data <-as.data.frame(matrix(nrow=nrow(precip),ncol=3))
colnames(data) <- c("precip","tempmin","tempmax")

# start with tempmin

par(mfrow=c(5,9))

for (i in sequence){
  
  data[,1] <- precip[,i]
  data[,2] <- tempmin[,i]
  data[,3] <- tempmax[,i]
  
  base <- 'Location '
  corr <- round(cor(data$precip,data$tempmin),2)
  fit <- lm(precip ~ tempmin, data)
  location <- paste(base,i,'; r=',corr,sep='')
  
  plot(tempmin[,i],precip[,i],
       xlab="Temperature [deg C]",
       ylab="Precipitation [mm]",
       main=location
  )
  abline(fit, col="red")
  
}

#### SPATIAL #####################################################
# Does precipitation correlate accorss stations?

n = nrow(precip) # iterate over 14975 observations

sequence = seq(from=1,to=n,by=1)

data <-as.data.frame(precip)

data[,1] <- precip[1,]
