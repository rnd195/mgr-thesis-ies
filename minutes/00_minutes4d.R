## --------------------
## Description: Minute data analysis - 4 days
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
## Package versions:
## aTSA 3.1.2
## forecast 8.21
## DescTools 0.99.48
## dplyr 1.0.10
## xts 0.13.0
## zoo 1.8.11
## librarian 1.8.1
## --------------------

#### Setup ####
if (!require(librarian)) install.packages("librarian")
librarian::shelf(xts, dplyr, rstudioapi, DescTools, forecast, aTSA)
setwd(dirname(getActiveDocumentContext()$path))
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")

series_raw <- readRDS("data/load_1min_2011_2021_xts.rds")

#### Pre-estimation ####
# See pre_estimation.R

series <- series_raw %>%
  log() %>%
  diff() %>%
  na.omit()

# Make sure that you're starting on 2011-01-01 00:00:00! (Central European Time)
head(series)

#### Prepare dates ####
starting <- seq.Date(as.Date("2011-01-01"), as.Date("2021-12-27"), by = "day") %>%
  as.character()
ending <- seq.Date(as.Date("2011-01-04"), as.Date("2021-12-30"), by = "day") %>%
  as.character()
oos_days <- seq.Date(as.Date("2011-01-05"), as.Date("2021-12-31"), by = "day") %>%
  as.character()
reg_days <- paste0(starting, "/", ending)


#### Run the models ####
cols <- c(
  "train_days", "oos_day", "spec", "fcst_rmse", "rw_rmse", "fcst_mae", "rw_mae", paste0("box_p_lag", 4:7), "jb_p", "dm_p_1", "dm_p_2"
)
results_df <- as.data.frame(matrix(nrow = length(reg_days), ncol = length(cols)))
colnames(results_df) <- cols

j <- 0
for (days in reg_days) {
  j <- j + 1
  cat("\nTraining days: ", days, "\nOOS day: ", oos_days[j], "\nProgress: ", j, "/", length(reg_days))
  # Autofind a model
  mod <- auto.arima(series[days], ic = "aic", max.p = 10, max.q = 10, max.d = 1, seasonal = F, approximation = F, stepwise = F)

  # If the lowest-AIC model is 'ARIMA(0,0,0) with zero mean', refit using a baseline
  if (sum(mod$arma) == 0) {
    mod <- Arima(series[days], order = c(1, 0, 1), include.mean = F)
  }

  # Out of sample fit
  onestepfit <- Arima(
    series[oos_days[j]],
    model = mod
  )
  # Generate one step ahead preds (https://stats.stackexchange.com/a/55197) and RW
  onesteps <- xts(fitted(onestepfit), order.by = index(series[oos_days[j]]))
  actuals <- series[oos_days[j]]
  rw_preds <- rep(0, length(actuals))

  # Add diagnostics
  boxtests <- rep(NA, 4)
  for (lag in 4:7) {
    boxtests[lag - 3] <- Box.test(mod$residuals, lag = lag, type = "Lj")$p.value
  }
  jbtest <- JarqueBeraTest(mod$residuals)$p.value

  # Diebold-Mariano test
  resid1 <- as.numeric(residuals(onestepfit))
  resid2 <- as.numeric(actuals - rw_preds)
  dmtest1 <- dm.test(
    resid1,
    resid2,
    # Alt hypothesis is that method 2 is less accurate than method 1
    alternative = "less",
    power = 1
  )$p.value
  dmtest2 <- dm.test(
    resid1,
    resid2,
    alternative = "less",
    power = 2
  )$p.value

  # Save results
  results_df[j, "train_days"] <- days
  results_df[j, "oos_day"] <- oos_days[j]
  results_df[j, "spec"] <- paste(mod)
  results_df[j, paste0("box_p_lag", 4:7)] <- boxtests
  results_df[j, "jb_p"] <- jbtest
  results_df[j, c("dm_p_1", "dm_p_2")] <- c(dmtest1, dmtest2)
  results_df[j, c("fcst_rmse", "rw_rmse", "fcst_mae", "rw_mae")] <- c(
    RMSE(onesteps, actuals),
    RMSE(rw_preds, actuals),
    MAE(onesteps, actuals),
    MAE(rw_preds, actuals)
  )
  # Write csv every 10 models
  if ((j %% 10) == 0) write.csv(results_df, "results_df_4d_v2.csv")
}
write.csv(results_df, "results_df_4d_v2.csv")


# Histogram
results_df <- na.omit(results_df)
fcst_rmse_hist <- hist(as.numeric(results_df$fcst_rmse), breaks = 50)
rw_rmse_hist <- hist(as.numeric(results_df$rw_rmse), breaks = 50)
plot(fcst_rmse_hist, col = rgb(0, 0, 1, 0.25))
plot(rw_rmse_hist, col = rgb(1, 0, 0, 0.25), add = T)

fcst_mae_hist <- hist(as.numeric(results_df$fcst_mae), breaks = 50)
rw_mae_hist <- hist(as.numeric(results_df$rw_mae), breaks = 50)
plot(fcst_mae_hist, col = rgb(0, 0, 1, 0.25))
plot(rw_mae_hist, col = rgb(1, 0, 0, 0.25), add = T)

# CDF
p1r <- ecdf(results_df$fcst_rmse)
p2r <- ecdf(results_df$rw_rmse)
p1m <- ecdf(results_df$fcst_mae)
p2m <- ecdf(results_df$rw_mae)

plot(p1r, col = rgb(0, 0, 1, 0.5))
plot(p2r, col = rgb(1, 0, 0, 0.5), add = T)

plot(p1m, col = rgb(0, 0, 1, 0.5))
plot(p2m, col = rgb(1, 0, 0, 0.5), add = T)

table(results_df$dm_p_1 < 0.05)
table(results_df$dm_p_2 < 0.05)
table(results_df$fcst_rmse < results_df$rw_rmse)
table(results_df$fcst_mae < results_df$rw_mae)
