# Initial repo population.
rm(list = ls())

## grab PRECIP data ##########
precip <- read.csv("SWAT Input Master Data Precipitation.csv") ## load in data

## grab TEMP data ##########
tempmin <- read.csv("SWAT Input Master Data Temperature - Min.csv") ## load in data
tempmax <- read.csv("SWAT Input Master Data Temperature - Max.csv") ## load in data

