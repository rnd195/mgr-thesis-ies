## --------------------
## Description: NN modeling and forecasting 48-hours-ahead
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
##
## Package versions:
## DescTools 0.99.48
## feasts 0.3.0
## fabletools 0.3.2
## torch 0.9.1
## lubridate 1.9.2
## forcats 1.0.0
## stringr 1.5.0
## purrr 1.0.1
## readr 2.1.4
## tidyr 1.3.0
## tibble 3.1.8
## ggplot2 3.4.1
## tidyverse 2.0.0
## rstudioapi 0.14
## dplyr 1.1.0
## xts 0.13.0
## zoo 1.8.11
## tsibble 1.1.3
## librarian 1.8.1
##
## Code adapted from:
## Keydana (2021). Posit AI Blog: Introductory time-series forecasting with torch. Retrieved from https://blogs.rstudio.com/tensorflow/posts/2021-03-10-forecasting-time-series-with-torch_1/
## Keydana (2021). Posit AI Blog: torch time series continued: A first go at multi-step prediction. Retrieved from https://blogs.rstudio.com/tensorflow/posts/2021-03-11-forecasting-time-series-with-torch_2/
## --------------------

#### Setup ####
if (!require(librarian)) install.packages("librarian")
librarian::shelf(tsibble, xts, dplyr, rstudioapi, tidyverse, torch, feasts, lubridate, DescTools)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))
seed <- 1111
set.seed(seed)
torch_manual_seed(seed)

##### Functions #####
# Process data in training and validation
train_batch <- function(bt) {
  optimizer$zero_grad()
  output <- net(bt$x$to(device = device))
  target <- bt$y$to(device = device)
  loss <- nnf_mse_loss(output, target)
  loss$backward()
  optimizer$step()
  loss$item()
}
valid_batch <- function(bt) {
  output <- net(bt$x$to(device = device))
  target <- bt$y$to(device = device)
  loss <- nnf_mse_loss(output, target)
  loss$item()
}

# Subset tsibble data based on date
subset_data <- function(start, end, vars = "load_mw") {
  all_data_ts %>%
    filter(Index >= as.Date(start), Index < as.Date(end) %m+% days(1)) %>%
    as_tibble() %>%
    select(all_of(vars))
}

##### Load data #####
# # Columns to difference (based on the ADF test from regression_trees_search.R)
# nonrejected_cols <- c(
#   "air_pressure_diff_t1", "price_eur_mwh_t1", "temperature_t1", "visibility_distance_t1",
#   "visibility_distance_diff_t1", "wind_speed_t1", "wind_speed_diff_t1", "load_mw_max_24_hrs",
#   "air_pressure_diff_t24", "air_pressure_diff_t48", "air_pressure_diff_t72", "air_pressure_diff_max_24_hrs",
#   "price_eur_mwh_t24", "price_eur_mwh_t48", "price_eur_mwh_t72", "price_eur_mwh_max_24_hrs",
#   "temperature_t24", "temperature_t48", "temperature_t72", "temperature_max_24_hrs", "temperature_diff_max_24_hrs",
#   "visibility_distance_t24", "visibility_distance_t48", "visibility_distance_t72", "visibility_distance_max_24_hrs",
#   "visibility_distance_diff_t24", "visibility_distance_diff_t48", "visibility_distance_diff_t72",
#   "visibility_distance_diff_max_24_hrs", "wind_speed_t24", "wind_speed_t48", "wind_speed_t72",
#   "wind_speed_max_24_hrs", "wind_speed_diff_t24", "wind_speed_diff_t48", "wind_speed_diff_t72",
#   "wind_speed_diff_max_24_hrs"
# )
# # Did not really help in out of sample forecasting => Not applied

# 1h load, weather, price and forecasts
series <- readRDS("data/series_features_1h_2012_2021_v2.rds")
dummies <- readRDS("data/seasonal_dummies_3.rds")

# series[, nonrejected_cols] <- series[, nonrejected_cols] %>%
#   diff()

all_data <- merge(series["2012/"], dummies["2012/"])
head(all_data)

#### Prepare data ####
# Convert data to tsibble format
all_data_ts <- all_data %>%
  fortify.zoo() %>%
  as_tsibble()

# Get dummy variable colnames for further usage
dummy_vars_names <- all_data_ts %>%
  as.data.frame() %>%
  select(mon_Feb:hour_23.wknd_or_h) %>%
  colnames()

variables <- c(
  "load_mw",
  "load_mw_t1",
  "load_mw_max_24_hrs",
  "load_mw_t24",
  "load_mw_avg_7_days",
  "load_mw_t72",
  "load_mw_t48",
  "temperature_avg_7_days",
  "temperature_t72",
  "temperature_t48",
  "temperature",
  "temperature_max_24_hrs",
  "temperature_t24",
  "temperature_t1",
  # "price_eur_mwh_t1",
  # "price_eur_mwh",
  # "price_eur_mwh_t24",
  dummy_vars_names
)
train_set <- c("2016-01-01", "2020-05-31")
valid_set <- c("2020-06-01", "2020-12-31")
test_set <- c("2021-01-01", "2021-12-31")

# Be careful with changing this parameter due to the intersection with the oos set
# -> only multiples of 24
n_timesteps <- 5 * 24

# 48 hours ahead
n_forecast <- 2 * 24

batch_size <- 128

# Change the start of the test_set so that, after the lookback parameter (n_timesteps)
# is considered, the predictions start at 2021-01-01 00:00
test_set[1] <- as.Date(test_set[1]) %m-% days(n_timesteps / 24) %>%
  as.character()

# Generate in-sample (train), validation, and out-of-sample (test) data
train_data <- subset_data(train_set[1], train_set[2], vars = variables) %>% as.matrix()
valid_data <- subset_data(valid_set[1], valid_set[2], vars = variables) %>% as.matrix()
test_data <- subset_data(test_set[1], test_set[2], vars = variables) %>% as.matrix()

# Save mean and sd for normalization and denormalization into a matrix
train_norm_df <- as.data.frame(matrix(nrow = length(variables), ncol = 2))
colnames(train_norm_df) <- c("mean", "sd")
rownames(train_norm_df) <- variables
for (col in variables) {
  train_norm_df[col, "mean"] <- mean(train_data[, col])
  train_norm_df[col, "sd"] <- sd(train_data[, col])
}
norm_mx <- as.matrix(train_norm_df)
head(norm_mx)

##### Prepare the data for torch #####
load_dataset <- dataset(
  name = "load_dataset",
  initialize = function(data, n_timesteps, n_forecast, sample_frac = 1) {

    # Normalize data (don't forget to add into self$x, and into the variables vector above)
    load_mw <- (data[, "load_mw"] - norm_mx["load_mw", 1]) / norm_mx["load_mw", 2]
    load_mw_t1 <- (data[, "load_mw_t1"] - norm_mx["load_mw_t1", 1]) / norm_mx["load_mw_t1", 2]
    load_mw_max_24_hrs <- (data[, "load_mw_max_24_hrs"] - norm_mx["load_mw_max_24_hrs", 1]) / norm_mx["load_mw_max_24_hrs", 2]
    load_mw_t24 <- (data[, "load_mw_t24"] - norm_mx["load_mw_t24", 1]) / norm_mx["load_mw_t24", 2]
    load_mw_avg_7_days <- (data[, "load_mw_avg_7_days"] - norm_mx["load_mw_avg_7_days", 1]) / norm_mx["load_mw_avg_7_days", 2]
    load_mw_t72 <- (data[, "load_mw_t72"] - norm_mx["load_mw_t72", 1]) / norm_mx["load_mw_t72", 2]
    load_mw_t48 <- (data[, "load_mw_t48"] - norm_mx["load_mw_t48", 1]) / norm_mx["load_mw_t48", 2]
    temperature_avg_7_days <- (data[, "temperature_avg_7_days"] - norm_mx["temperature_avg_7_days", 1]) / norm_mx["temperature_avg_7_days", 2]
    temperature_t72 <- (data[, "temperature_t72"] - norm_mx["temperature_t72", 1]) / norm_mx["temperature_t72", 2]
    temperature_t48 <- (data[, "temperature_t48"] - norm_mx["temperature_t48", 1]) / norm_mx["temperature_t48", 2]
    temperature <- (data[, "temperature"] - norm_mx["temperature", 1]) / norm_mx["temperature", 2]
    temperature_max_24_hrs <- (data[, "temperature_max_24_hrs"] - norm_mx["temperature_max_24_hrs", 1]) / norm_mx["temperature_max_24_hrs", 2]
    temperature_t24 <- (data[, "temperature_t24"] - norm_mx["temperature_t24", 1]) / norm_mx["temperature_t24", 2]
    temperature_t1 <- (data[, "temperature_t1"] - norm_mx["temperature_t1", 1]) / norm_mx["temperature_t1", 2]
    # price_eur_mwh_t1 <- (data[, "price_eur_mwh_t1"] - norm_mx["price_eur_mwh_t1", 1]) / norm_mx["price_eur_mwh_t1", 2]
    # price_eur_mwh <- (data[, "price_eur_mwh"] - norm_mx["price_eur_mwh", 1]) / norm_mx["price_eur_mwh", 2]
    # price_eur_mwh_t24 <- (data[, "price_eur_mwh_t24"] - norm_mx["price_eur_mwh_t24", 1]) / norm_mx["price_eur_mwh_t24", 2]

    # Dummies not normalized
    dummies <- data[, dummy_vars_names]

    # Convert to torch tensor and transform to a correct form
    # Ensure that load is in the first position
    self$x <- cbind(
      load_mw,
      load_mw_t1,
      load_mw_max_24_hrs,
      load_mw_t24,
      load_mw_avg_7_days,
      load_mw_t72,
      load_mw_t48,
      temperature_avg_7_days,
      temperature_t72,
      temperature_t48,
      temperature,
      temperature_max_24_hrs,
      temperature_t24,
      temperature_t1,
      # price_eur_mwh_t1,
      # price_eur_mwh,
      # price_eur_mwh_t24,
      dummies
      ) %>%
      torch_tensor()
    # Save timesteps and forecast length
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    n <- nrow(self$x) - self$n_timesteps - self$n_forecast + 1
    self$starts <- sort(
      sample.int(
        n = n,
        size = n * sample_frac
      )
    )
  },
  .getitem = function(i) {
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1
    pred_length <- self$n_forecast
    list(
      x = self$x[start:end, ],
      y = self$x[(end + 1):(end + pred_length), 1]
    )
  },
  .length = function() {
    length(self$starts)
  }
)

# Apply the dataset creation function to our data
train_ds <- load_dataset(train_data, n_timesteps, n_forecast, sample_frac = 0.5)
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)

valid_ds <- load_dataset(valid_data, n_timesteps, n_forecast, sample_frac = 0.5)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)

test_ds <- load_dataset(test_data, n_timesteps, n_forecast)
test_dl <- test_ds %>% dataloader(batch_size = 1)


#### NN specification ####
model_lstm <- nn_module(
  initialize = function(input_size, hidden_size, linear_size, output_size, num_layers = 1, dropout = 0, linear_dropout = 0) {
    self$num_layers <- num_layers
    self$linear_dropout <- linear_dropout
    self$rnn <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = dropout,
      batch_first = TRUE
    )
    self$mlp <- nn_sequential(
      nn_linear(hidden_size, linear_size),
      nn_relu(),
      nn_dropout(linear_dropout),
      nn_linear(linear_size, output_size)
    )
  },
  forward = function(x) {
    x <- self$rnn(x)
    x[[1]][, -1, ..] %>%
      self$mlp()
  }
)

# Set NN parameters
net <- model_lstm(
  input_size = length(variables),
  hidden_size = 128,
  linear_size = 512,
  num_layers = 1,
  output_size = n_forecast,
  dropout = 0,
  linear_dropout = 0
)
net

#### Fit ####
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
net <- net$to(device = device)

# Specify the number of epochs and the optimizer
num_epochs <- 17
optimizer <- optim_adam(net$parameters, lr = 0.001)

# Prepare training and validation scheme
train_batch <- function(b) {
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_mse_loss(output, target)
  loss$backward()
  optimizer$step()
  loss$item()
}

valid_batch <- function(b) {
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  loss <- nnf_mse_loss(output, target)
  loss$item()
}

# Perform training
valid_losses <- rep(NA, length(num_epochs))
train_losses <- rep(NA, length(num_epochs))

for (epoch in 1:num_epochs) {
  net$train()
  train_loss <- c()
  coro::loop(for (b in train_dl) {
    loss <- train_batch(b)
    train_loss <- c(train_loss, loss)
  })
  train_losses[epoch] <- mean(train_loss)
  cat(sprintf("\nEpoch %d, training loss: %3.5f\n", epoch, mean(train_loss)))
  net$eval()
  valid_loss <- c()
  coro::loop(for (b in valid_dl) {
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
  })
  valid_losses[epoch] <- mean(valid_loss)
  cat(sprintf("Epoch %d, validation loss: %3.5f\n", epoch, mean(valid_loss)))
}

plot(train_losses, type = "l")
lines(valid_losses, col = "red")
abline(v = 17, col = "grey")

#### Generate forecasts ####
# Load the model if needed
# net <- torch_load("weights/nn_48h_2017.gzip")

##### Out of sample #####
net$eval()
# Allocate an empty list of predictions
test_preds <- vector(mode = "list", length = length(test_dl))

# Fill this list with predictions
i <- 1
coro::loop(for (b in test_dl) {
  input <- b$x
  output <- net(input$to(device = device))
  preds <- as.numeric(output)
  test_preds[[i]] <- preds
  i <<- i + 1
})

##### Validation #####
# Borrow 5 days of data from the in-sample set due to the n_timesteps parameter
valid_set2_start <- as.Date(valid_set[1]) %m-% days(n_timesteps / 24) %>%
  as.character()
valid_data2 <- subset_data(valid_set2_start, valid_set[2], vars = variables) %>% as.matrix()
valid_ds2 <- load_dataset(valid_data2, n_timesteps, n_forecast)
valid_dl2 <- valid_ds2 %>% dataloader(batch_size = 1)

# Generate the preds
valid_preds <- vector(mode = "list", length = length(valid_dl2))
k <- 1
coro::loop(for (bv in valid_dl2) {
  input_v <- bv$x
  output_v <- net(input_v$to(device = device))
  preds_v <- as.numeric(output_v)
  valid_preds[[k]] <- preds_v
  k <<- k + 1
})

# Because these are multistep predictions, I need to create correct
# indices to extract the correct predictions. Thus, I start by
# recreating the validation set as a tsibble object as it contains
# the index
valid_data_indices <- subset_data(valid_set[1], valid_set[2], vars = "Index") %>%
  filter(Index >= as.Date("2020-06-01"))
# I then add 1:(length of validation) as a new column
valid_data_indices$Index_num <- 1:nrow(valid_data_indices)
# And leave every 24th of these indices
valid_indices_all <- valid_data_indices %>%
  as_tibble() %>%
  filter(hour(Index) == 23) %>%
  select(Index_num) %>%
  unlist() %>%
  as.numeric()
# Because this is 48 steps ahead, I filter out every second of these daily
# indices to obtain bi-daily indices
valid_indices_dynamic <- valid_indices_all[c(F, T)] + 1
# And I drop the last index because it's 47 hours instead of 48h
valid_indices_dynamic[length(valid_indices_dynamic)] <- NA
valid_indices_dynamic <- na.omit(valid_indices_dynamic)

# I now extract the predictions into their own table
fcst_ts_48_v <- na.omit(data.frame(fcst = NA))
for (l in c(1, valid_indices_dynamic)) {
  valid_48 <- valid_preds[[l]]
  # Add NAs as padding to make sure you grab the correct prediction
  valid_48 <- c(
    rep(NA, n_timesteps + (l - 1)),
    valid_48,
    rep(NA, nrow(valid_data2) - (l - 1) - n_timesteps - n_forecast)
  ) %>% na.omit()
  # Make sure to de-standardize the data
  fcst_ts_48_v <- rbind(
    fcst_ts_48_v,
    data.frame(fcst = valid_48 * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"])
  )
}

# Remove one specific observation due to time change
valid_data_indices[3600:3605, ]

valid_df <- data.frame(
  load_mw = valid_data[, "load_mw"],
  fcst = fcst_ts_48_v[-3602, ]
)

# Evaluate
DescTools::RMSE(valid_df$fcst, valid_df$load_mw)
DescTools::MAPE(valid_df$fcst, valid_df$load_mw)

##### In-sample ####
# Recreate the in-sample set
train_set2_start <- as.Date(train_set[1]) %m-% days(n_timesteps / 24) %>%
  as.character()
train_data2 <- subset_data(train_set2_start, train_set[2], vars = variables) %>% as.matrix()
train_ds2 <- load_dataset(train_data2, n_timesteps, n_forecast)
train_dl2 <- train_ds2 %>% dataloader(batch_size = 1)

# Generate the fitted values
train_preds <- vector(mode = "list", length = length(train_dl2))
k <- 1
coro::loop(for (bt in train_dl2) {
  input_t <- bt$x
  output_t <- net(input_t$to(device = device))
  preds_t <- as.numeric(output_t)
  train_preds[[k]] <- preds_t
  k <<- k + 1
})

# Extract the predictions to a table (fitted values will be one-step-ahead in-sample predictions)
fcst_ts_onestep_t <- data.frame(fcst = rep(NA, nrow(train_data2)))
for (m in 1:length(train_preds)) {
  train_onestep <- train_preds[[m]][1]
  # Add NAs as padding to make sure you grab the correct prediction
  train_onestep <- c(
    rep(NA, n_timesteps + (m - 1)),
    train_onestep,
    rep(NA, nrow(train_data2) - (m - 1) - n_timesteps - 1)
  )
  # Make sure to de-standardize the data
  fcst_ts_onestep_t[m, "fcst"] <- na.omit(
    train_onestep * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"]
  )
}
train_df <- data.frame(
  load_mw = train_data[1:nrow(na.omit(fcst_ts_onestep_t)), "load_mw"],
  fit = na.omit(fcst_ts_onestep_t)
)
train_xts <- xts(
  train_df,
  order.by = index(series[paste0(train_set[1], "/", train_set[2])])[1:nrow(train_df)]
)
# Save fitted values
# saveRDS(train_xts, "results/insample/nn48_fit.rds")

##### Two days ahead #####
oos_data <- subset_data(test_set[1], test_set[2], vars = c("Index", "load_mw", "load_mw_f"))
## You can test that it, indeed, outputs predictions two days ahead:
## change the oos set to end on 2020-12-31, and observe that test_pred1
## is the same as when the full oos set is used.
# oos_data <- subset_data(test_set[1], "2020-12-31", vars = c("Index", "load_mw", "load_mw_f"))

# Create the first table, add the rest
test_pred1 <- test_preds[[1]]
test_pred1 <- c(
  rep(NA, n_timesteps + 0),
  test_pred1,
  rep(NA, nrow(oos_data) - 0 - n_timesteps - n_forecast)
)
fcst_ts <- oos_data %>%
  add_column(
    fcst = test_pred1 * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"]
  ) %>%
  na.omit()

# Again, the second set of predictions has index 48+1, the next one 2*48+1 ...
# 2066 - 2067 skips from 01 to 03 due to time change
# and 2021-03-29 00:00:00 doesn't exist for some reason
oos_data_indices <- oos_data %>%
  filter(Index >= as.Date("2021-01-01"))
oos_data_indices$Index_num <- 1:nrow(oos_data_indices)
indices_all <- oos_data_indices %>%
  as_tibble() %>%
  filter(hour(Index) == 23) %>%
  select(Index_num) %>%
  unlist() %>%
  as.numeric()

indices_dynamic <- indices_all[c(F, T)] + 1
# Drop the last one because it's 24 hours instead of 48h
indices_dynamic[length(indices_dynamic)] <- NA
indices_dynamic <- na.omit(indices_dynamic)

## Alternatively, use Hard-coded index (don't)
# indices <- seq(49, length(test_preds), by = 48)

# Assign predictions to indices for 48 ahead predictions
for (i in indices_dynamic) {
  test_pred <- test_preds[[i]]
  test_pred <- c(
    rep(NA, n_timesteps + (i - 1)),
    test_pred,
    rep(NA, nrow(oos_data) - (i - 1) - n_timesteps - n_forecast)
  )
  fcst_ts_add <- oos_data %>%
    add_column(
      fcst = test_pred * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"]
    ) %>%
    na.omit()
  fcst_ts <- rbind(fcst_ts, fcst_ts_add)
}

##### Evaluate 48h #####
# RMSE and MAPE by month
fcst_ts %>%
  group_by(month(Index)) %>%
  summarize(
    rmse_ofic = mean(DescTools::RMSE(load_mw, load_mw_f)),
    rmse_nn = mean(DescTools::RMSE(load_mw, fcst)),
    mape_ofic = mean(DescTools::MAPE(load_mw, load_mw_f)),
    mape_nn = mean(DescTools::MAPE(load_mw, fcst)),
  )

# Convert to xts object
fcsts_xts <- fcst_ts %>%
  select(-Index) %>%
  na.omit() %>%
  xts(order.by = na.omit(fcst_ts)$Index)

# Sample plots
plot(fcsts_xts["2021-01-01/2021-01-10"], legend.loc = "topright")
plot(fcsts_xts["2021-04-20/2021-04-24"], legend.loc = "topright")
plot(fcsts_xts["2021-07-10/2021-07-16"], legend.loc = "topright")
plot(fcsts_xts["2021-10-25/2021-10-29"], legend.loc = "topright")

# Total OOS metrics
DescTools::RMSE(fcst_ts$fcst, fcst_ts$load_mw)
DescTools::MAPE(fcst_ts$fcst, fcst_ts$load_mw)

# Comparison with the official forecasts
DescTools::RMSE(fcst_ts$load_mw, fcst_ts$load_mw_f)
DescTools::MAPE(fcst_ts$load_mw, fcst_ts$load_mw_f)

# Workdays only
fcsts_xts %>%
  merge(dummies["2021"]$wknd_or_h) %>%
  fortify.zoo() %>%
  filter(wknd_or_h == 0) %>%
  group_by(month = month(Index)) %>%
  summarize(
    rmse_ofic = mean(DescTools::RMSE(load_mw, load_mw_f)),
    rmse_nn = mean(DescTools::RMSE(load_mw, fcst)),
    mape_ofic = mean(DescTools::MAPE(load_mw, load_mw_f)),
    mape_nn = mean(DescTools::MAPE(load_mw, fcst)),
  )

# colnames(fcsts_xts) <- c("load_mw", "ofic", "nn")
# saveRDS(fcsts_xts, "all_forecasts_48_nn.rds")
