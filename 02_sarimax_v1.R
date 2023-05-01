## --------------------
## Description: SARIMAX modeling and forecasting (version 1)
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
## lubridate 1.9.2
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, lubridate, forecast, DescTools, purrr, rstudioapi)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))

# 1h load, weather, price and forecasts
series <- readRDS("data/series_1h_2011_2021_xts.rds")["2012/"]
# Seasonal dummies
dummy_vars <- readRDS("data/seasonal_dummies_3.rds")["2012/"]

#### Functions ####
# In case errors would happen
possibly_Arima <- possibly(
  .f = Arima,
  otherwise = "err"
)

#### Setup dates ####
# Sets of data - train on the in-sample, find best parameters on the valid
# sample, and make forecasts on the out of sample period
in_sample <- "2018-01-01/2020-05-31"
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

#### Run model ####
mod1 <- possibly_Arima(
  series$load_mw[in_sample],
  order = c(2, 0, 2),
  seasonal = list(order = c(1, 0, 1), period = 24),
  xreg = as.matrix(dummy_vars[in_sample])
)

# In-sample fit
in_sample_fit <- data.frame(fit = fitted(mod1), load_mw = series$load_mw[in_sample]) %>%
  xts(order.by = index(series$load_mw[in_sample]))
# saveRDS(in_sample_fit, "results/insample/sarimax101_fit.rds")
# saveRDS(in_sample_fit, "results/insample/sarimax202_fit.rds")

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
one_step_MAPE_valid <- DescTools::MAPE(
  series$load_mw[valid_sample],
  as.numeric(one_step_ahead_valid)
)
one_step_RMSE_valid <- DescTools::RMSE(
  series$load_mw[valid_sample],
  as.numeric(one_step_ahead_valid)
)

# One step ahead forecasts on the oos set
onestepfit_oos <- Arima(
  series$load_mw[oos_period],
  model = mod1,
  xreg = as.matrix(dummy_vars[oos_period])
)
fits_oos_xts <- xts(
  fitted(onestepfit_oos),
  order.by = zoo::index(series$load_mw[oos_period])
)
one_step_ahead_oos <- fits_oos_xts[oos_period]
one_step_MAPE_oos <- DescTools::MAPE(
  series$load_mw[oos_period],
  as.numeric(one_step_ahead_oos)
)
one_step_RMSE_oos <- DescTools::RMSE(
  series$load_mw[oos_period],
  as.numeric(one_step_ahead_oos)
)

##### 48h ahead #####
# Validation
valid_period <- paste0(valid_starting[1], "/", valid_ending[length(valid_ending)])
valid_preds_48 <- xts(
  rep(NA, nrow(series[valid_period])),
  order.by = zoo::index(series[valid_period])
)

for (i in 1:length(valid_xreg_dates)) {
  # For each 48-hour periods, save actual values for comparison
  # No need to save the official forecasts in the validation set
  actuals <- as.numeric(series$load_mw[valid_xreg_dates[i]])
  # Input the new data into the old model
  newfit <- Arima(
    series$load_mw[valid_expand_date[i]],
    model = mod1,
    xreg = as.matrix(dummy_vars[valid_expand_date[i]])
  )
  # The length of the sample is the same as the nrow of xreg == 48
  sarimax_fcst <- forecast(newfit, xreg = as.matrix(dummy_vars[valid_xreg_dates[i]]))
  # Save forecasts
  valid_preds_48[valid_xreg_dates[i]] <- as.numeric(sarimax_fcst$mean)
}

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
valid_error

# Out of sample
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
oos_error

#### Save all info ####
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
results <- as.data.frame(matrix(ncol = length(cols), nrow = 1))
colnames(results) <- cols
results[1, c("RMSE_insample", "MAPE_insample")] <- c(
  DescTools::RMSE(as.numeric(series$load_mw[in_sample]), as.numeric(fitted(mod1))),
  DescTools::MAPE(as.numeric(series$load_mw[in_sample]), as.numeric(fitted(mod1)))
)
results[1, c("in_sample", "spec")] <- c(in_sample, paste(mod1))
results[1, c(
  "RMSE_valid_me",
  "MAPE_valid_me"
)] <- valid_error
results[1, c(
  "RMSE_oos_me",
  "MAPE_oos_me",
  "RMSE_oos_ofic",
  "MAPE_oos_ofic"
)] <- oos_error
results[1, c("one_step_RMSE_valid", "one_step_MAPE_valid")] <- c(one_step_RMSE_valid, one_step_MAPE_valid)
results[1, c("one_step_RMSE_oos", "one_step_MAPE_oos")] <- c(one_step_RMSE_oos, one_step_MAPE_oos)

# Print results in a nicer format
as.data.frame(t(results))

results %>%
  select(RMSE_oos_me:MAPE_oos_ofic)

#### Comparisons ####
series_oos <- series[oos_period][, c("load_mw", "load_mw_f")]

# 48 h ahead
series_oos$two_days_f <- oos_preds_48h

plot(series_oos["2021-01-01/2021-01-10"][, c("load_mw", "load_mw_f", "two_days_f")], legend.loc = "topright")
plot(series_oos["2021-04-20/2021-04-24"][, c("load_mw", "load_mw_f", "two_days_f")], legend.loc = "topright")
plot(series_oos["2021-07-10/2021-07-16"][, c("load_mw", "load_mw_f", "two_days_f")], legend.loc = "topright")
plot(series_oos["2021-10-25/2021-10-29"][, c("load_mw", "load_mw_f", "two_days_f")], legend.loc = "topright")

# RMSE and MAPE by month
series_oos %>%
  fortify.zoo() %>%
  group_by(month = month(Index)) %>%
  summarize(
    rmse_me = mean(DescTools::RMSE(load_mw, two_days_f)),
    mape_me = mean(DescTools::MAPE(load_mw, two_days_f)),
    rmse_ofic = mean(DescTools::RMSE(load_mw, load_mw_f)),
    mape_ofic = mean(DescTools::MAPE(load_mw, load_mw_f))
  )

# One step ahead
series_oos$one_step_f <- xts(fits_oos_xts, order.by = zoo::index(series[oos_period]))
# By month
series_oos %>%
  fortify.zoo() %>%
  group_by(month = month(Index)) %>%
  summarize(
    # Only compare against actual
    rmse_onestep = mean(DescTools::RMSE(load_mw, one_step_f)),
    mape_onestep = mean(DescTools::MAPE(load_mw, one_step_f)),
  )

# By hour
series_oos %>%
  fortify.zoo() %>%
  group_by(hour = hour(Index)) %>%
  summarize(
    # Only compare against actual
    rmse_onestep = mean(DescTools::RMSE(load_mw, one_step_f)),
    mape_onestep = mean(DescTools::MAPE(load_mw, one_step_f)),
  )

#### Diagnostics ####
# TODO REDO
Acf(mod1$residuals)
Pacf(mod1$residuals)

# L-B test - H0: No serial correlation
checkresiduals(mod1)
Box.test(residuals(mod1), type = "Ljung-Box", lag = 1)
Box.test(residuals(mod1), type = "Ljung-Box", lag = 2)
Box.test(residuals(mod1), type = "Ljung-Box", lag = 3)
Box.test(residuals(mod1), type = "Ljung-Box", lag = 10)
Box.test(residuals(mod1), type = "Ljung-Box", lag = 11)
Box.test(residuals(mod1), type = "Ljung-Box", lag = 12)

# H0: Normally distributed
JarqueBeraTest(residuals(mod1))

# H0: No ARCH effects
# Todo


# Save preds 48h
# colnames(oos_preds_48h) <- "sarimax"
# saveRDS(oos_preds_48h, "all_forecasts_48_s.rds")
# Save preds 1h
# colnames(fits_oos_xts) <- "sarimax"
# saveRDS(fits_oos_xts, "all_forecasts_1_s.rds")

paste(mod1)
coefficients <- cbind(round(mod1$coef, 5), round(sqrt(diag(vcov(mod1))), 5))
colnames(coefficients) <- c("coef", "se")
write.csv(coefficients, "results/sarimax_coefs.csv")
