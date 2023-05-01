## --------------------
## Description: Creating dummy variable sets
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
##
## Package versions:
## dplyr 1.1.0
## xts 0.13.0
## rstudioapi 0.14
## lubridate 1.9.2
## readxl 1.4.2
## mlr 2.19.1
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, lubridate, mlr, readxl, rstudioapi)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))

# 1h load, weather, price and forecasts
series <- readRDS("data/series_1h_2011_2021_xts.rds")

#### Seasonal dummies ####
# First, define the general dummies
day_of_the_week <- format(index(series), "day_%a") %>%
  factor(levels = c("day_Mon", "day_Tue", "day_Wed", "day_Thu", "day_Fri", "day_Sat", "day_Sun"))
month_of_the_year <- format(index(series), "mon_%b") %>%
  factor(levels = c("mon_Jan", "mon_Feb", "mon_Mar", "mon_Apr", "mon_May", "mon_Jun", "mon_Jul", "mon_Aug", "mon_Sep", "mon_Oct", "mon_Nov", "mon_Dec"))
hour_of_day <- format(index(series), "hour_%H")

# holidays
holidays_xlsx <- read_excel("data/holidays_2011_2021.xlsx")
holidays_xlsx$date <- force_tz(holidays_xlsx$date, tzone = "CET")
date_short <- format(index(series), "%Y-%m-%d")
holidays_vector <- ifelse(
  date_short %in% format(holidays_xlsx$date, "%Y-%m-%d"),
  1,
  0
)
holidays_xts <- xts(holidays_vector, order.by = index(series))

# workdays vs nonworking days (holidays and weekends)
week_and_weekends <- ifelse(
  day_of_the_week %in% c("day_Sat", "day_Sun"),
  "wknd_or_h",
  as.character(day_of_the_week)
)
wd_df <- data.frame(
  date = index(series),
  date_short = date_short,
  wknds_hlds = week_and_weekends
)

wd_df[date_short %in% format(holidays_xlsx$date, "%Y-%m-%d"), ]$wknds_hlds <- "wknd_or_h"

# Covid dummy variable with respect to the states of emergency in CZE
# https://www.seznamzpravy.cz/clanek/fakta-velky-prehled-dva-roky-s-koronavirem-v-cesku-190958
emergency1 <- seq.Date(as.Date("2020-03-12"), as.Date("2020-05-17"), by = "day")
emergency2 <- seq.Date(as.Date("2020-10-05"), as.Date("2021-04-11"), by = "day")
emergency3 <- seq.Date(as.Date("2021-11-26"), as.Date("2021-12-25"), by = "day")

wd_df$covid_dummy <- 0
wd_df[wd_df$date_short %in% as.character(c(emergency1, emergency2, emergency3)), ]$covid_dummy <- 1
covid_dummy <- xts(wd_df$covid_dummy, order.by = index(series))

##### Seasonal dummies 1 #####
# Month, day, hour, holidays, covid
seasonal_dummies_1 <- merge(
  xts(
    cbind(
      createDummyFeatures(month_of_the_year, method = "reference"),
      createDummyFeatures(day_of_the_week, method = "reference"),
      createDummyFeatures(hour_of_day, method = "reference")
    ),
    order.by = index(series)
  ),
  holidays_xts,
  covid_dummy
)

##### Seasonal dummies 2 #####
# Month, working/nonworking day, hour, covid
seasonal_dummies_2 <- merge(
  xts(
    cbind(
      createDummyFeatures(month_of_the_year, method = "reference"),
      createDummyFeatures(wd_df$wknds_hlds, method = "reference"),
      createDummyFeatures(hour_of_day, method = "reference")
    ),
    order.by = index(series)
  ),
  covid_dummy
)

##### Seasonal dummies 3 #####
# Like 2 but with hourly interactions for nonworking days

hour_interactions_nonworking <- model.matrix(
  ~ (hour_01 + hour_02 + hour_03 + hour_04 + hour_05 + hour_06 +
       hour_07 + hour_08 + hour_09 + hour_10 + hour_11 + hour_12 +
       hour_13 + hour_14 + hour_15 + hour_16 + hour_17 + hour_18 +
       hour_19 + hour_20 + hour_21 + hour_22 + hour_23) * wknd_or_h,
  seasonal_dummies_2
)

seasonal_dummies_3 <- merge(
  seasonal_dummies_2[, setdiff(colnames(seasonal_dummies_2), colnames(hour_interactions_nonworking))],
  hour_interactions_nonworking[, -1]
)
