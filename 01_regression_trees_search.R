## --------------------
## Description: Bagged regression trees grid search
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
## Package versions:
## aTSA 3.1.2
## ipred 0.9.13
## rpart 4.1.19
## rstudioapi 0.14
## DescTools 0.99.48
## xts 0.13.0
## zoo 1.8.11
## dplyr 1.1.0
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, DescTools, rstudioapi, rpart, ipred, aTSA)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))
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
in_sample1 <- "2017-06-01/2020-05-31"
in_sample2 <- "2017-01-01/2020-05-31"
in_sample3 <- "2016-06-01/2020-05-31"
in_sample4 <- "2016-01-01/2020-05-31"
in_sample5 <- "2015-06-01/2020-05-31"

valid_sample <- "2020-06-01/2020-12-31"
oos_period <- "2021-01-01/2021-12-30"

#### Grid search ####
# https://petolau.github.io/Regression-trees-for-forecasting-time-series-in-R/
# https://uc-r.github.io/regression_trees

hyper_grid <- expand.grid(
  minsplit = seq(18, 30, 1),
  maxdepth = seq(16, 22, 1),
  complex_param = seq(1e-08, 1e-06, length.out = 5),
  # in_sample_sets = c(in_sample1, in_sample2, in_sample3)
  in_sample_sets = c(in_sample2)
)
cat(nrow(hyper_grid), "specifications will be estimated.\n")

# Define data samples
in_sample <- in_sample2
df_in_sample <- as.data.frame(all[in_sample])
df_valid <- as.data.frame(all[valid_sample])
df_oos <- as.data.frame(all[oos_period])

for (i in 1:nrow(hyper_grid)) {
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  complex_param <- hyper_grid$complex_param[i]
  nbag <- hyper_grid$nbag[i]
  # in_sample <- as.character(hyper_grid$in_sample_sets[i])

  tree <- bagging(
    load_mw ~ .,
    data = df_in_sample,
    nbagg = 30,
    control = rpart.control(
      maxdepth = maxdepth,
      minsplit = minsplit,
      cp = complex_param
    )
  )

  # In sample
  hyper_grid[i, "in_sample_rmse"] <- DescTools::RMSE(
    predict(tree, df_in_sample),
    all[in_sample]$load_mw
  )
  # Validation
  valid_preds <- predict(tree, df_valid)
  hyper_grid[i, "valid_rmse"] <- DescTools::RMSE(
    valid_preds,
    all[valid_sample]$load_mw
  )
  hyper_grid[i, "valid_mape"] <- DescTools::MAPE(
    valid_preds,
    all[valid_sample]$load_mw
  )
  # Out of sample
  oos_preds <- predict(tree, df_oos)
  hyper_grid[i, "oos_rmse"] <- DescTools::RMSE(
    oos_preds,
    all[oos_period]$load_mw
  )
  hyper_grid[i, "oos_mape"] <- DescTools::MAPE(
    oos_preds,
    all[oos_period]$load_mw
  )

  # For printing results in a notebook
  # flush.console()
  # Write the csv file every iteration
  write.csv(na.omit(hyper_grid), "trees_hyper_grid_v3.csv")
  cat(i, "/", nrow(hyper_grid), "\n")
  # Partially clear memory
  rm(tree)
}


# # Illustrative example used in the thesis
# tree_example <- rpart(
#   load_mw ~ temperature_t1,
#   data = series[in_sample2],
# )
# library(rpart.plot)
# rpart.plot(tree_example)
# test_pred <- data.frame(temperature_t1 = 15)
# predict(tree_example, test_pred)
