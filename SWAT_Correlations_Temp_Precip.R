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

#### ANNUAL #####################################################
n <- ncol(precip) # number of total locations

# Configure:
loc <- 14 # pick a location from 1 to n
startyear <- 1980 # provide starting year for dataset

# initialize dataframes
alldata <-as.data.frame(matrix(nrow=nrow(precip),ncol=3)) # daily
colnames(alldata) <- c("precip","tempmin","tempmax")

monthlydata <-as.data.frame(matrix(nrow=12,ncol=3)) # monthly
colnames(monthlydata) <- c("precip","tempmin","tempmax")


totalyears <- floor(nrow(precip)/365)
years <- seq(from=startyear, to=startyear+totalyears-1,by=1)
year <- 999 # initialize

# select data
# 14,"port14pcp",41.4375,-83.3125,192.4
# Woodville Township, OH
alldata$precip <- precip[,loc]
alldata$tempmax <- tempmax[,loc]
alldata$tempmin <- tempmin[,loc]

averagedata <- as.data.frame(matrix(nrow=12,ncol=3)) # average annual
averagedata[,] <- 0
colnames(averagedata) <- c("precip","tempmin","tempmax")

# BEGIN MEGAPLOT
par(mfrow=c(5,9))

for (i in seq(from=1, to=length(years), by=1)){
  # BEGIN LOOP
  
  # find in and out dates
  year<-years[i]
  inday <- 1+365*(year-startyear)
  outday <- inday+364

  data <-alldata # initialize
  data <- as.data.frame(alldata[inday:outday,])
  
  # Provide a summary of average monthly data
  # january - 31 days (1 to 31)
  monthlydata$precip[1] <- mean(data$precip[1:31])
  monthlydata$tempmin[1] <- mean(data$tempmin[1:31])
  monthlydata$tempmax[1] <- mean(data$tempmax[1:31])
  
  # february - 28 days (32 to 59)
  monthlydata$precip[2] <- mean(data$precip[32:59])
  monthlydata$tempmin[2] <- mean(data$tempmin[32:59])
  monthlydata$tempmax[2] <- mean(data$tempmax[32:59])
  
  # March - 31 days (60 to 90)
  monthlydata$precip[3] <- mean(data$precip[60:90])
  monthlydata$tempmin[3] <- mean(data$tempmin[60:90])
  monthlydata$tempmax[3] <- mean(data$tempmax[60:90])
  
  # April - 30 days (91 to 120)
  monthlydata$precip[4] <- mean(data$precip[91:120])
  monthlydata$tempmin[4] <- mean(data$tempmin[91:120])
  monthlydata$tempmax[4] <- mean(data$tempmax[91:120])
  
  # May - 31 days (121 to 151)
  monthlydata$precip[5] <- mean(data$precip[121:151])
  monthlydata$tempmin[5] <- mean(data$tempmin[121:151])
  monthlydata$tempmax[5] <- mean(data$tempmax[121:151])
  
  # June - 30 days (152 to 181)
  monthlydata$precip[6] <- mean(data$precip[152:181])
  monthlydata$tempmin[6] <- mean(data$tempmin[152:181])
  monthlydata$tempmax[6] <- mean(data$tempmax[152:181])
  
  # July - 31 days (182 to 212)
  monthlydata$precip[7] <- mean(data$precip[182:212])
  monthlydata$tempmin[7] <- mean(data$tempmin[182:212])
  monthlydata$tempmax[7] <- mean(data$tempmax[182:212])
  
  # August - 31 days (213 to 243)
  monthlydata$precip[8] <- mean(data$precip[213:243])
  monthlydata$tempmin[8] <- mean(data$tempmin[213:243])
  monthlydata$tempmax[8] <- mean(data$tempmax[213:243])
  
  # September - 30 days (244 to 273)
  monthlydata$precip[9] <- mean(data$precip[244:273])
  monthlydata$tempmin[9] <- mean(data$tempmin[244:273])
  monthlydata$tempmax[9] <- mean(data$tempmax[244:273])
  
  # October - 31 days (274 to 304)
  monthlydata$precip[10] <- mean(data$precip[274:304])
  monthlydata$tempmin[10] <- mean(data$tempmin[274:304])
  monthlydata$tempmax[10] <- mean(data$tempmax[274:304])
  
  # November - 30 days (305 to 334)
  monthlydata$precip[11] <- mean(data$precip[305:334])
  monthlydata$tempmin[11] <- mean(data$tempmin[305:334])
  monthlydata$tempmax[11] <- mean(data$tempmax[305:334])
  
  # December - 31 days (335 to 365)
  monthlydata$precip[12] <- mean(data$precip[335:365])
  monthlydata$tempmin[12] <- mean(data$tempmin[335:365])
  monthlydata$tempmax[12] <- mean(data$tempmax[335:365])
  
  # Convert to Kelvin
  monthlydata$tempmin <- monthlydata$tempmin+273.15
  monthlydata$tempmax <- monthlydata$tempmax+273.15
  
  # compute annual average
  averagedata$precip <- averagedata$precip + monthlydata$precip
  averagedata$tempmax <- averagedata$tempmax + monthlydata$tempmax
  averagedata$tempmin <- averagedata$tempmin + monthlydata$tempmin
  
  # PLOT
  par(mar=c(2, # bottom
            4, # left 
            2, # top 
            3  # right
            ) + 0.1)
  min <- 0
  max <- 8
  df.bar <- barplot(height=monthlydata$precip, col="lightblue",
                    ylim=c(min,max),
                    ylab="Precipitation [mm]",
                    main=year)
  
  monthlabels <- c("Jan", "Feb", "Mar", "Apr",
                   "May", "Jun", "Jul", "Aug",
                   "Sep", "Oct", "Nov", "Dec")
  
  axis(1, at=df.bar,labels=monthlabels)
  
  # normalize temperatures
  monthlydata_scaled <- monthlydata
  tempmax <- max(alldata$tempmax)+273.15
  tempmin <- min(alldata$tempmin)+273.15
  monthlydata_scaled$tempmin <- (monthlydata$tempmin-tempmin)/(tempmax-tempmin)*max
  monthlydata_scaled$tempmax <- (monthlydata$tempmax-tempmin)/(tempmax-tempmin)*max
  
  tempaxis <- (seq(from=tempmin, to=tempmax, length=8)-tempmin)/(tempmax-tempmin)*max
  
  axis(4, at=tempaxis, labels=floor(seq(from=tempmin, to=tempmax,length=8)))
  # mtext("Temperature [K]", side=4, line=3, las=2,cex.lab=0.5)
  
  points(x=df.bar,monthlydata_scaled$tempmax,
         xaxt="n",
         pch=19,
         col="red")
  
  
  lines(x=df.bar,monthlydata_scaled$tempmax,
        lwd=2,
        col="red")
  points(x=df.bar,monthlydata_scaled$tempmin,
         pch=19,
         col="blue"
  )
  lines(x=df.bar,monthlydata_scaled$tempmin,
        lwd=2,
        col="blue"
  )
  
}

plot(1, type = "n", axes=FALSE, xlab="", ylab=""
     # ,main="Global mean sea level at 2100 [m]"
)
legend(x = "top",inset = 0, 
       # legend = c("Wide priors combined with data (high-temperature scenario)" , 
       #            "Wide priors combined with data and expert assessment (high-temperature scenario)",
       #            "17th-83rd percentile - SEJ2018 expert assessment (high-temperature scenario)",
       #            "5-95th percentile - SEF2018 expert assessment (high-temperature scenario)"
       #            # "IPCC AR6 likely range (RCP 8.5)"
       # ),
       
       # legend = c("BRICK" , 
       #            "BRICK + SEJ2018",
       #            expression("5"^th~"-95"^th~" percentile, SEJ2018"),
       #            expression("17"^th~"-83"^rd~" percentile, SEJ2018")
       #            # "IPCC AR6 likely range (RCP 8.5)"
       # ), 
       legend = c("Temp Max [K]", "Temp Min [K]"), 
       pch = c(19, 19), 
       lwd = c(2,2),
       col = c("red","blue")
       
)


#### AVERAGE ANNUAL #####################################################
# compute average
averagedata[,] <- averagedata[,]/length(years)

# PLOT
par(mar=c(2, # bottom
          4, # left 
          2, # top 
          3  # right
) + 0.1)
min <- 0
max <- 8
df.bar <- barplot(height=averagedata$precip, col="lightblue",
                  ylim=c(min,max),
                  ylab="Precipitation [mm]",
                  main="1980-2020 Average, Woodville Township, OH")

monthlabels <- c("Jan", "Feb", "Mar", "Apr",
                 "May", "Jun", "Jul", "Aug",
                 "Sep", "Oct", "Nov", "Dec")

axis(1, at=df.bar,labels=monthlabels)

# normalize temperatures
averagedata_scaled <- averagedata
tempmax <- max(alldata$tempmax)+273.15
tempmin <- min(alldata$tempmin)+273.15
averagedata_scaled$tempmin <- (averagedata$tempmin-tempmin)/(tempmax-tempmin)*max
averagedata_scaled$tempmax <- (averagedata$tempmax-tempmin)/(tempmax-tempmin)*max

tempaxis <- (seq(from=tempmin, to=tempmax, length=8)-tempmin)/(tempmax-tempmin)*max

axis(4, at=tempaxis, labels=floor(seq(from=tempmin, to=tempmax,length=8)))
# mtext("Temperature [K]", side=4, line=3, las=2,cex.lab=0.5)

points(x=df.bar,monthlydata_scaled$tempmax,
       xaxt="n",
       pch=19,
       col="red")


lines(x=df.bar,monthlydata_scaled$tempmax,
      lwd=2,
      col="red")
points(x=df.bar,monthlydata_scaled$tempmin,
       pch=19,
       col="blue"
)
lines(x=df.bar,monthlydata_scaled$tempmin,
      lwd=2,
      col="blue"
)

legend( "topright",
legend = c("Temp Max [K]", "Temp Min [K]"), 
pch = c(19, 19), 
lwd = c(2,2),
col = c("red","blue")
)

