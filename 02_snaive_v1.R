## --------------------
## Description: Seasonal naive model (version 1)
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
##
## Package versions:
## dplyr 1.1.0
## xts 0.13.0
## forecast 8.21
## DescTools 0.99.48
## rstudioapi 0.14
## lubridate 1.9.2
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, DescTools, rstudioapi, forecast, lubridate)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))

# Load data
series <- readRDS("data/series_features_1h_2012_2021_v2.rds")$load_mw

# Convert to msts and align the seasonal periods perfectly with the previous year
series_ts <- msts(series["2019/2020"]$load_mw, seasonal.periods = c(24, 24 * 7, 24 * 363.9))
head(series_ts)

# Produce snaive forecasts for 2021
snaive_f <- snaive(series_ts, h = 8758)$mean

series_2021 <- series["2021"] %>%
  merge(snaive_f)

# Sample plots
plot(series_2021["2021-01-01/2021-01-10"], legend.loc = "topright")
plot(series_2021["2021-04-20/2021-04-24"], legend.loc = "topright")
plot(series_2021["2021-08-10/2021-08-16"], legend.loc = "topright")
plot(series_2021["2021-10-25/2021-10-29"], legend.loc = "topright")

# Total RMSE and MAPE
DescTools::RMSE(series_2021$load_mw, series_2021$snaive_f)
DescTools::MAPE(series_2021$load_mw, series_2021$snaive_f)

# RMSE and MAPE by month
series_2021 %>%
  fortify.zoo() %>%
  group_by(month = month(Index)) %>%
  summarize(
    # Only compare against actual
    rmse = mean(DescTools::RMSE(load_mw, snaive_f)),
    mape = mean(DescTools::MAPE(load_mw, snaive_f)),
  )

# Save preds
# colnames(series_2021) <- c("actual", "snaive")
# saveRDS(series_2021$snaive, "all_forecasts_48_snaive.rds")

# Add validation set SNAIVE for completeness
series_ts2 <- msts(series["2018/2019"]$load_mw, seasonal.periods = c(24, 24 * 7, 24 * 363.9))
snaive_f2 <- snaive(series_ts2, h = 8781)$mean
ts2 <- xts(snaive_f2, order.by = index(series["2020"]))

DescTools::RMSE(ts2["2020-06-01/2020-12-31"], series["2020-06-01/2020-12-30"]$load_mw)
DescTools::MAPE(ts2["2020-06-01/2020-12-31"], series["2020-06-01/2020-12-30"]$load_mw)
