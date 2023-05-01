## --------------------
## Description: Analysis of minute forecasts
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
##
## Package versions:
## dplyr 1.1.0
## xts 0.13.0
## DescTools 0.99.48
## rstudioapi 0.14
## ggplot2 3.4.1
## lubridate 1.9.2
## forecast 8.21
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(xts, dplyr, DescTools, rstudioapi, ggplot2, lubridate, forecast)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))

# Load data
forecasts <- readRDS("data/results_fcst_2d_v3.rds")
series <- readRDS("data/load_1min_2011_2021_xts.rds") %>%
  log() %>%
  diff()

series_forecasts <- merge(series[zoo::index(forecasts)], forecasts)
rw_forecasts <- rep(0, nrow(series_forecasts))

# Load results
results2d <- read.csv("results_df_2d_v3.csv")
# I added the ARCH test later, so these two result tables only lack that, hence v2
results3d <- read.csv("results_df_3d_v2.csv")
results4d <- read.csv("results_df_4d_v2.csv")

#### Reporting ####
# Mean RMSE and MAE for each out-of-sample day
data.frame(mean(results2d$fcst_rmse), mean(results3d$fcst_rmse), mean(results4d$fcst_rmse))
data.frame(mean(results2d$fcst_mae), mean(results3d$fcst_mae), mean(results4d$fcst_mae))
# Median RMSE and MAE for each out-of-sample day
data.frame(median(results2d$fcst_rmse), median(results3d$fcst_rmse), median(results4d$fcst_rmse))
data.frame(median(results2d$fcst_mae), median(results3d$fcst_mae), median(results4d$fcst_mae))

# Choose the most accurate of the three
results <- results2d

# Which models were the most common?
results$spec <- as.factor(results$spec)
results %>%
  select(spec) %>%
  table() %>%
  sort() %>%
  tail(10) %>%
  as.data.frame() %>%
  ggplot(aes(x = spec, y = Freq)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_bw()

# More accurate forecasts in absolute terms
table(results$fcst_rmse < results$rw_rmse)
table(results$fcst_mae < results$rw_mae)

# Histograms
ggplot(results) +
  geom_histogram(aes(x = fcst_rmse), bins = 50, position = "identity", fill = "blue", alpha = 0.35) +
  geom_histogram(aes(x = rw_rmse), bins = 50, position = "identity", fill = "red", alpha = 0.35) +
  theme_bw() +
  labs(title = "RMSE") +
  xlim(0.002, 0.008)

ggplot(results) +
  geom_histogram(aes(x = fcst_mae), bins = 50, position = "identity", fill = "blue", alpha = 0.35) +
  geom_histogram(aes(x = rw_mae), bins = 50, position = "identity", fill = "red", alpha = 0.35) +
  theme_bw() +
  labs(title = "MAE")

# Empirical CDF
p1r <- ecdf(results$fcst_rmse)
p2r <- ecdf(results$rw_rmse)
p1m <- ecdf(results$fcst_mae)
p2m <- ecdf(results$rw_mae)

plot(p1r, col = rgb(0, 0, 1, 0.5), xlim = c(0.002, 0.006))
plot(p2r, col = rgb(1, 0, 0, 0.5), xlim = c(0.002, 0.006), add = T)

plot(p1m, col = rgb(0, 0, 1, 0.5))
plot(p2m, col = rgb(1, 0, 0, 0.5), add = T)

# DM tests, H_Alt: RW forecasts are less accurate than ARIMA forecasts
dm.test(
  as.numeric(series_forecasts$load_mw - series_forecasts$forecast),
  as.numeric(series_forecasts$load_mw - rw_forecasts),
  # Alt hypothesis is that method 2 is less accurate than method 1
  alternative = "less",
  power = 1
)
dm.test(
  as.numeric(series_forecasts$load_mw - series_forecasts$forecast),
  as.numeric(series_forecasts$load_mw - rw_forecasts),
  # Alt hypothesis is that method 2 is less accurate than method 1
  alternative = "less",
  power = 2
)

# Per day
table(results$dm_p_1 < 0.05)
table(results$dm_p_2 < 0.05)

# Ljung Box tests, H_0: no serial correlation
table(results$box_p_lag4 < 0.05)
table(results$box_p_lag5 < 0.05)
table(results$box_p_lag6 < 0.05)
table(results$box_p_lag7 < 0.05)

# ARCH tests, H_0: no ARCH effects in the residuals
table(results$arch_p_lag4 < 0.05)
table(results$arch_p_lag5 < 0.05)
table(results$arch_p_lag6 < 0.05)
table(results$arch_p_lag7 < 0.05)


# Jarque Bera test, H_0: normally distributed
table(results$jb_p < 0.05)

# Which months typically have lower RMSE for RW?
results %>%
  arrange(rw_rmse) %>%
  head(50)

results$year <- year(results$oos_day)
results %>%
  group_by(year) %>%
  summarize(
    fcst_year_mean_rmse = mean(fcst_rmse),
    fcst_year_mean_mae = mean(fcst_mae),
    rw_year_mean_rmse = mean(rw_rmse),
    rw_year_mean_mae = mean(rw_mae)
  )
