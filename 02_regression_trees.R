## --------------------
## Description: Bagged regression trees modeling and forecasting one-step-ahead
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
##
## Package versions:
## dplyr 1.1.0
## xts 0.13.0
## DescTools 0.99.48
## rpart 4.1.19
## caret 6.0.93
## rpart.plot 3.1.1
## ipred 0.9.13
## lubridate 1.9.2
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, DescTools, rstudioapi, rpart, caret, rpart.plot, ipred, lubridate)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))
# Set seed for reproducibility due to bagging
seed <- 1111
set.seed(seed)

# Load, weather, price, official forecasts, and constructed features
series <- readRDS("data/series_features_1h_2012_2021_v2.rds")
# Constructed dummy variables
dummies <- readRDS("data/seasonal_dummies_3.rds")["2012/"]

#### Lag variables ####
# Original variables
base_variables <- head(series) %>%
  as.data.frame() %>%
  select(load_mw:wind_speed_diff) %>%
  colnames()
# 1 lag variables (no need to lag)
lag_cols <- colnames(series)[grep("*_t[0-9]*", colnames(series))]
no_lag <- c(
  lag_cols,
  base_variables
)
# Lag all variables except for dummies and official fcst
series_lag <- dplyr::lag(series[, setdiff(colnames(series), no_lag)]) %>%
  merge(series[, c("load_mw", "load_mw_f", lag_cols)], .)

# # Perform adf tests on a subset of data for differencing
# adfs <- data.frame(
#   type1 = rep(NA, ncol(series_lag)),
#   type2 = rep(NA, ncol(series_lag)),
#   type3 = rep(NA, ncol(series_lag))
# )
# rownames(adfs) <- colnames(series_lag)
# i <- 0
# for (col in colnames(series_lag)) {
#   i <- i + 1
#   adf <- aTSA::adf.test(series_lag["2017/2020"][, col], nlag = 10, output = F)
#   adfs[i, "type1"] <- sum(adf$type1[, "p.value"] < 0.05)
#   adfs[i, "type2"] <- sum(adf$type2[, "p.value"] < 0.05)
#   adfs[i, "type3"] <- sum(adf$type3[, "p.value"] < 0.05)
#   cat(i, "/", ncol(series_lag), "\n")
# }
#
# # Where null was never rejected (=> nonstationarity)
# nonrejected <- adfs[with(adfs, (type1 + type2 + type3 == 0)), ]
# nonrejected_cols <- rownames(nonrejected)

# Copy the columns so that it doesn't have to run every time
nonrejected_cols <- c(
  "air_pressure_diff_t1", "price_eur_mwh_t1", "temperature_t1", "visibility_distance_t1",
  "visibility_distance_diff_t1", "wind_speed_t1", "wind_speed_diff_t1", "load_mw_max_24_hrs",
  "air_pressure_diff_t24", "air_pressure_diff_t48", "air_pressure_diff_t72", "air_pressure_diff_max_24_hrs",
  "price_eur_mwh_t24", "price_eur_mwh_t48", "price_eur_mwh_t72", "price_eur_mwh_max_24_hrs",
  "temperature_t24", "temperature_t48", "temperature_t72", "temperature_max_24_hrs", "temperature_diff_max_24_hrs",
  "visibility_distance_t24", "visibility_distance_t48", "visibility_distance_t72", "visibility_distance_max_24_hrs",
  "visibility_distance_diff_t24", "visibility_distance_diff_t48", "visibility_distance_diff_t72",
  "visibility_distance_diff_max_24_hrs", "wind_speed_t24", "wind_speed_t48", "wind_speed_t72",
  "wind_speed_max_24_hrs", "wind_speed_diff_t24", "wind_speed_diff_t48", "wind_speed_diff_t72",
  "wind_speed_diff_max_24_hrs"
)

# Picked variables for first differences
series_lag[, nonrejected_cols] <- series_lag[, nonrejected_cols] %>%
  diff()

# Add last hour's diff
series_lag$load_mw_t1_diff <- dplyr::lag(diff(series_lag$load_mw))

# Some of the correlations are high, but let the bagged trees handle it
round(cor(na.omit(series_lag)), 3)

# Remove official forecast
cols <- setdiff(colnames(series_lag), "load_mw_f")
all <- merge(series_lag[, cols]["2012/"], dummies)

#### Prepare dates ####
# Try three different in-sample sets
in_sample1 <- "2012-01-01/2020-05-31"
in_sample2 <- "2013-01-01/2020-05-31"
in_sample3 <- "2014-01-01/2020-05-31"
in_sample4 <- "2015-01-01/2020-05-31"
in_sample5 <- "2016-01-01/2020-05-31"
in_sample6 <- "2017-01-01/2020-05-31"

valid_sample <- "2020-06-01/2020-12-31"
oos_period <- "2021-01-01/2021-12-30"

#### Model ####

# Define data samples
in_sample <- in_sample2
df_in_sample <- as.data.frame(all[in_sample])
df_valid <- as.data.frame(all[valid_sample])
df_oos <- as.data.frame(all[oos_period])

best_spec <- data.frame(
  in_sample_set = in_sample,
  in_sample_rmse = NA,
  valid_rmse = NA,
  valid_mape = NA,
  oos_rmse = NA,
  oos_mape = NA
)

tree <- bagging(
  load_mw ~ .,
  data = df_in_sample,
  nbagg = 30,
  control = rpart.control(
    maxdepth = 21,
    minsplit = 18,
    cp = 0.0000002575
  )
)

in_sample_fit <- data.frame(
  fit = predict(tree, df_in_sample),
  load_mw = all[in_sample]$load_mw
) %>%
  xts(order.by = index(all[in_sample]$load_mw))
# saveRDS(in_sample_fit, "results/insample/trees_fit.rds")

# In sample
best_spec$in_sample_rmse <- DescTools::RMSE(
  in_sample_fit$fit,
  all[in_sample]$load_mw
)

# Validation
valid_preds <- predict(tree, df_valid)
best_spec$valid_rmse <- DescTools::RMSE(
  valid_preds,
  all[valid_sample]$load_mw
)
best_spec$valid_mape <- DescTools::MAPE(
  valid_preds,
  all[valid_sample]$load_mw
)

# Out of sample
oos_preds <- predict(tree, df_oos)
best_spec$oos_rmse <- DescTools::RMSE(
  oos_preds,
  all[oos_period]$load_mw
)
best_spec$oos_mape <- DescTools::MAPE(
  oos_preds,
  all[oos_period]$load_mw
)


#### Evaluation ####
best_spec

oos_preds <- predict(tree, df_oos)
oos_xts <- data.frame(actuals = df_oos$load_mw, oos_preds = oos_preds) %>%
  xts(order.by = index(all[oos_period]))

head(oos_xts)

plot(oos_xts["2021-01-01/2021-01-10"])
plot(oos_xts["2021-06-10/2021-06-20"])
plot(oos_xts["2021-09-06/2021-09-15"])
plot(oos_xts["2021-12-20/2021-12-29"])

# By month
oos_xts %>%
  fortify.zoo() %>%
  group_by(month = month(Index)) %>%
  summarize(
    # Only compare actual vs NN
    rmse_tree = mean(DescTools::RMSE(actuals, oos_preds)),
    mape_tree = mean(DescTools::MAPE(actuals, oos_preds)),
  )

# By hour
oos_xts %>%
  fortify.zoo() %>%
  group_by(hour = hour(Index)) %>%
  summarize(
    # Only compare actual vs NN
    rmse_tree = mean(DescTools::RMSE(actuals, oos_preds)),
    mape_tree = mean(DescTools::MAPE(actuals, oos_preds)),
  )

# Variable importance (total SSR decrease across bagged trees)
var_importance <- varImp(tree)
top_ten <- var_importance %>%
  arrange(desc(Overall)) %>%
  head(10)
top_ten

# Save model
# saveRDS(tree, file = "tree21-17_2013.rds")

# Save preds
# colnames(oos_xts) <- c("actuals", "tree")
# saveRDS(oos_xts[, 2], "all_forecasts_1_tree.rds")
