## --------------------
## Description: SARIMAX grid search (version 1)
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
## purrr 1.0.1
## rstudioapi 0.14
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, forecast, DescTools, purrr, rstudioapi)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))

# 1h load, weather, price, and forecasts
series <- readRDS("data/series_1h_2011_2021_xts.rds")["2012/"]
# Seasonal dummies
dummy_vars <- readRDS("data/seasonal_dummies_3.rds")["2012/"]

#### Preparation ####
# Dataframe for saving the results
cols <- c(
  "in_sample",
  "spec",
  "RMSE_insample",
  "MAPE_insample",
  "RMSE_valid_me",
  "MAPE_valid_me",
  "RMSE_oos_me",
  "MAPE_oos_me",
  "RMSE_oos_ofic",
  "MAPE_oos_ofic",
  "one_step_RMSE_valid",
  "one_step_MAPE_valid",
  "one_step_RMSE_oos",
  "one_step_MAPE_oos"
)
results <- as.data.frame(matrix(ncol = length(cols), nrow = 999))
colnames(results) <- cols

# In case errors happen, output "err" instead of stopping computation
possibly_Arima <- possibly(
  .f = Arima,
  otherwise = "err"
)

#### Setup dates and grid ####
# Sets of data - train on the in-sample, find best parameters on the valid
# sample, and make forecasts on the out of sample period
in_sample <- "2017-01-01/2020-05-31"
valid_sample <- "2020-06-01/2020-12-31"
oos_period <- "2021-01-01/2021-12-30"

valid_starting <- seq.Date(as.Date("2020-06-01"), as.Date("2020-12-30"), by = "day")
valid_ending <- seq.Date(as.Date("2020-06-02"), as.Date("2020-12-31"), by = "day")
# Sequence of 2 non-overlapping days
valid_xreg_dates <- paste0(valid_starting[c(T, F)], "/", valid_ending[c(T, F)])
# Expand the data so that predictions in the future can be made using past X values
valid_expand_end <- seq.Date(as.Date("2020-05-31"), as.Date("2020-12-29"), by = "day")[c(T, F)]
valid_expand_date <- paste0(substr(in_sample, 1, 10), "/", valid_expand_end)

# End on the 30th of Dec because of 48h increments
oos_starting <- seq.Date(as.Date("2021-01-01"), as.Date("2021-12-29"), by = "day")
oos_ending <- seq.Date(as.Date("2021-01-02"), as.Date("2021-12-30"), by = "day")
oos_xreg_dates <- paste0(oos_starting[c(T, F)], "/", oos_ending[c(T, F)])
oos_expand_end <- seq.Date(as.Date("2020-12-31"), as.Date("2021-12-28"), by = "day")[c(T, F)]
oos_expand_date <- paste0(substr(in_sample, 1, 10), "/", oos_expand_end)


# Hyperparameter grid
hyper_grid <- expand.grid(
  p = 1:4,
  d = 0:1,
  # q = 1:4,
  bigP = 0:1,
  bigQ = 0:1,
  bigD = 0:1
)

#### Run models ####
# The 'qs' parameter specifies the MA order
# Provides a possibility of simple parallelization of the grid search
# For example, by running a search for each q in a different R session
qs <- 1
save_csv <- paste0("q", qs, "sarimax.csv")

for (i in 1:nrow(hyper_grid)) {
  # Initialize params
  p <- hyper_grid[i, "p"]
  d <- hyper_grid[i, "d"]
  # q <- hyper_grid[i, "q"]
  bigP <- hyper_grid[i, "bigP"]
  bigD <- hyper_grid[i, "bigD"]
  bigQ <- hyper_grid[i, "bigQ"]

	# Train the model
  mod1 <- possibly_Arima(
    series$load_mw[in_sample],
    order = c(p, d, qs),
    seasonal = list(order = c(bigP, bigD, bigQ), period = 24),
    xreg = as.matrix(dummy_vars[in_sample])
  )
  cat("Model", i, "trained\n")

  # Skip iteration in case of problems and return a line of errors
  if ((mod1 == "err")[1]) {
  	results[i, ] <- rep("Error", ncol(results))
  	write.csv(na.omit(results), save_csv)
  	next
  }

  ##### One step ahead #####
  # One step ahead forecasts on the validation set
  onestepfit_valid <- Arima(
    series$load_mw[valid_sample],
    model = mod1,
    xreg = as.matrix(dummy_vars[valid_sample])
  )
  fits_valid <- fitted(onestepfit_valid)
  # Convert to xts for easier date manipulation
  fits_valid_xts <- xts(fits_valid, order.by = zoo::index(series$load_mw[valid_sample]))
  one_step_ahead_valid <- fits_valid_xts[valid_sample]
  # Calculate and save MAPE and RMSE
  one_step_mape_valid <- DescTools::MAPE(
    series$load_mw[valid_sample],
    as.numeric(one_step_ahead_valid)
  )
  one_step_rmse_valid <- DescTools::RMSE(
    series$load_mw[valid_sample],
    as.numeric(one_step_ahead_valid)
  )
  cat("One step validation", i, "\n")

  # One step ahead forecasts on the oos set
  onestepfit_oos <- Arima(
    series$load_mw[oos_period],
    model = mod1,
    xreg = as.matrix(dummy_vars[oos_period])
  )
  fits_oos <- fitted(onestepfit_oos)
  fits_oos_xts <- xts(fits_oos, order.by = zoo::index(series$load_mw[oos_period]))
  one_step_ahead_oos <- fits_oos_xts[oos_period]
  one_step_mape_oos <- DescTools::MAPE(
    series$load_mw[oos_period],
    as.numeric(one_step_ahead_oos)
  )
  one_step_rmse_oos <- DescTools::RMSE(
    series$load_mw[oos_period],
    as.numeric(one_step_ahead_oos)
  )
  cat("One step OOS", i, "\n")

  ##### 48h ahead #####
  # Validation
  valid_period <- paste0(valid_starting[1], "/", valid_ending[length(valid_ending)])
  valid_preds_48 <- xts(
    rep(NA, nrow(series[valid_period])),
    order.by = zoo::index(series[valid_period])
  )

  for (m in 1:length(valid_xreg_dates)) {
    # For each 48-hour periods, save actual values for comparison
    # No need to save the official forecasts in the validation set
    actuals <- as.numeric(series$load_mw[valid_xreg_dates[m]])
    # Input the new data and use the coefficients of the old model
    newfit <- Arima(
      series$load_mw[valid_expand_date[m]],
      # Coefficients from the already fitted model!
      model = mod1,
      xreg = as.matrix(dummy_vars[valid_expand_date[m]])
    )
    # The length of the forecast (48) is taken from xreg
    sarimax_fcst <- forecast(newfit, xreg = as.matrix(dummy_vars[valid_xreg_dates[m]]))
    # Save forecasts
    valid_preds_48[valid_xreg_dates[m]] <- as.numeric(sarimax_fcst$mean)
  }
  cat("48h ahead validation", i, "\n")

  valid_error <- data.frame(
    RMSE_valid_me = DescTools::RMSE(
      as.numeric(valid_preds_48),
      as.numeric(series[valid_period]$load_mw)
    ),
    MAPE_valid_me = DescTools::MAPE(
      as.numeric(valid_preds_48),
      as.numeric(series[valid_period]$load_mw)
    )
  )

  # Out of sample - analogous to the validation set
  oos_preds_48h <- xts(
    rep(NA, nrow(series[oos_period])),
    order.by = zoo::index(series[oos_period])
  )

  for (j in 1:length(oos_xreg_dates)) {
    # Create actual values as well as official values
    actuals <- as.numeric(series$load_mw[oos_xreg_dates[j]])

    newfit <- Arima(
      series$load_mw[oos_expand_date[j]],
      model = mod1,
      xreg = as.matrix(dummy_vars[oos_expand_date[j]])
    )

    sarimax_fcst <- forecast(newfit, xreg = as.matrix(dummy_vars[oos_xreg_dates[j]]))
    # Save the predictions to their own vector
    oos_preds_48h[oos_xreg_dates[j]] <- as.numeric(sarimax_fcst$mean)
  }
  oos_error <- data.frame(
    RMSE_oos_me = DescTools::RMSE(
      as.numeric(oos_preds_48h),
      as.numeric(series[oos_period]$load_mw)
    ),
    MAPE_oos_me = DescTools::MAPE(
      as.numeric(oos_preds_48h),
      as.numeric(series[oos_period]$load_mw)
    ),
    RMSE_oos_ofic = DescTools::RMSE(
      as.numeric(series[oos_period]$load_mw_f),
      as.numeric(series[oos_period]$load_mw)
    ),
    MAPE_oos_ofic = DescTools::MAPE(
      as.numeric(series[oos_period]$load_mw_f),
      as.numeric(series[oos_period]$load_mw)
    )
  )
  cat("48h OOS", i, "\n")

  #### Save all info ####
  results[i, c("RMSE_insample", "MAPE_insample")] <- c(
    DescTools::RMSE(as.numeric(series$load_mw[in_sample]), as.numeric(fitted(mod1))),
    DescTools::MAPE(as.numeric(series$load_mw[in_sample]), as.numeric(fitted(mod1)))
  )
  results[i, c("in_sample", "spec")] <- c(in_sample, paste(mod1))
  results[i, c(
    "RMSE_valid_me",
    "MAPE_valid_me"
  )] <- valid_error
  results[i, c(
    "RMSE_oos_me",
    "MAPE_oos_me",
    "RMSE_oos_ofic",
    "MAPE_oos_ofic"
  )] <- oos_error
  results[i, c("one_step_RMSE_valid", "one_step_MAPE_valid")] <- c(one_step_rmse_valid, one_step_mape_valid)
  results[i, c("one_step_RMSE_oos", "one_step_MAPE_oos")] <- c(one_step_rmse_oos, one_step_mape_oos)
  # Rewrite CSVs on every iteration (each specification takes a long time)
  write.csv(na.omit(results), save_csv)
  cat(i, "/", nrow(hyper_grid), "computed\n")
}
