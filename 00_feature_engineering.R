## --------------------
## Description: Creating new predictors using hourly data
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
##
## Package versions:
## dplyr 1.1.0
## xts 0.13.0
## rstudioapi 0.14
## librarian 1.8.1
##
## --------------------

#### Setup ####
if (!require(librarian)) install.packages("librarian")
librarian::shelf(xts, dplyr, rstudioapi)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))

series <- readRDS("data/series_1h_2011_2021_xts.rds")
dummies <- readRDS("data/seasonal_dummies_3.rds")

#### Feature engineering ####
# Inspired by Fan and Hyndman (2012) - lags, average in 7 days, and max in 24h
cols <- setdiff(colnames(series), "load_mw_f")

lagnames <- rep(NA, 6 * length(cols))
j <- 1
for (i in cols) {
  j <- j + 6
  lagnames[(j - 6):(j - 1)] <- c(
    paste0(i, "_t1"),
    paste0(i, "_t24"),
    paste0(i, "_t48"),
    paste0(i, "_t72"),
    paste0(i, "_avg_7_days"),
    paste0(i, "_max_24_hrs")
  )
}
new_vars <- data.frame(
  matrix(
    nrow = nrow(series),
    ncol = length(lagnames)
  )
)
colnames(new_vars) <- lagnames
series_df <- cbind(as.data.frame(series), new_vars)

for (var in cols) {
  series_df[, paste0(var, "_t1")] <- dplyr::lag(series_df[, var])
  series_df[, paste0(var, "_t24")] <- dplyr::lag(series_df[, var], 24)
  series_df[, paste0(var, "_t48")] <- dplyr::lag(series_df[, var], 48)
  series_df[, paste0(var, "_t72")] <- dplyr::lag(series_df[, var], 72)
  # Last week's average
  series_df[, paste0(var, "_avg_7_days")] <- rollmeanr(series_df[, var], 168, fill = NA)
  # Max in 24 hours
  series_df[, paste0(var, "_max_24_hrs")] <- rollmax(series_df[, var], 24, fill = NA)
}

series <- xts(series_df, order.by = index(series))

# Check correlation with load
cor(na.omit(series["2012/"]))[1, ]
colnames(series)[abs(cor(na.omit(series["2012/"])))[1, ] > 0.25]

# Save data
# saveRDS(series["2011-02-01/"], "data/series_features_1h_2012_2021_v2.rds")
# series_features_df <- as.data.frame(series["2011-02-01/"])
# cols_series_features <- colnames(series_features_df)
# series_features_df$date <- as.POSIXct(index(series_features))
# series_features_df <- series_features_df[, c("date", cols_series_features)]
# write_parquet(series_features_df, "data/series_features_1h_2012_2021.parquet")

# Print summary statistics of a few predictors (not lags)
series["2012/"][, grep("*24_hrs|*7_days", colnames(series))] %>%
  na.omit() %>%
  summary()
