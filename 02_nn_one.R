## --------------------
## Description: NN modeling and forecasting one-step-ahead
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
# Set fixed seed for reproducibility
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
train_set <- c("2015-01-01", "2020-05-31")
valid_set <- c("2020-06-01", "2020-12-31")
test_set <- c("2021-01-01", "2021-12-31")

# Be careful with changing this parameter due to the intersection with the oos set
# -> only multiples of 24
n_timesteps <- 5 * 24

# 1 hour ahead
n_forecast <- 1

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
  hidden_size = 192,
  linear_size = 256,
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
num_epochs <- 22
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

plot(train_losses, type = "l", ylim = c(0, 0.015))
lines(valid_losses, col = "red")
abline(v = 22, col = "grey")

#### Generate forecasts ####
# Load the model if needed
# net <- torch_load("weights/nn_1h_2015.gzip")

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

# Extract the predictions to a table
fcst_ts_onestep_v <- data.frame(fcst = rep(NA, nrow(valid_data2)))
for (l in 1:length(valid_preds)) {
  valid_onestep <- valid_preds[[l]][1]
  # Add NAs as padding to make sure you grab the correct prediction
  valid_onestep <- c(
    rep(NA, n_timesteps + (l - 1)),
    valid_onestep,
    rep(NA, nrow(valid_data2) - (l - 1) - n_timesteps - 1)
  )
  # Make sure to de-standardize the data
  fcst_ts_onestep_v[l, "fcst"] <- na.omit(
    valid_onestep * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"]
  )
}
valid_df <- data.frame(load_mw = valid_data[, "load_mw"], fcst = na.omit(fcst_ts_onestep_v))

# Evaluate
DescTools::RMSE(valid_df$fcst, valid_df$load_mw)
DescTools::MAPE(valid_df$fcst, valid_df$load_mw)

##### In-sample ####
train_set2_start <- as.Date(train_set[1]) %m-% days(n_timesteps / 24) %>%
  as.character()
train_data2 <- subset_data(train_set2_start, train_set[2], vars = variables) %>% as.matrix()
train_ds2 <- load_dataset(train_data2, n_timesteps, n_forecast)
train_dl2 <- train_ds2 %>% dataloader(batch_size = 1)

# Generate the preds
train_preds <- vector(mode = "list", length = length(train_dl2))
k <- 1
coro::loop(for (bt in train_dl2) {
  input_t <- bt$x
  output_t <- net(input_t$to(device = device))
  preds_t <- as.numeric(output_t)
  train_preds[[k]] <- preds_t
  k <<- k + 1
})

# Extract the predictions to a table
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
train_df <- data.frame(load_mw = train_data[, "load_mw"], fit = na.omit(fcst_ts_onestep_t))
train_xts <- xts(train_df, order.by = index(series[paste0(train_set[1], "/", train_set[2])]))
# Save fitted values
# saveRDS(train_xts, "results/insample/nn1_fit.rds")

##### One step ahead ####
oos_data <- subset_data(test_set[1], test_set[2], vars = c("Index", "load_mw", "load_mw_f"))

# Initialize the one step ahead predictions
test_pred1_onestep <- test_preds[[1]][1]
test_pred1_onestep <- c(
  rep(NA, n_timesteps + 0),
  test_pred1_onestep,
  rep(NA, nrow(oos_data) - 0 - n_timesteps - 1)
)
fcst_ts_onestep <- oos_data %>%
  add_column(
    fcst = test_pred1_onestep * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"]
  ) %>%
  na.omit()

# Takes a while
for (i in 2:length(test_preds)) {
  test_pred_onestep <- test_preds[[i]][1]
  test_pred_onestep <- c(
    rep(NA, n_timesteps + (i - 1)),
    test_pred_onestep,
    rep(NA, nrow(oos_data) - (i - 1) - n_timesteps - 1)
  )
  fcst_ts_onestep_add <- oos_data %>%
    add_column(
      fcst = test_pred_onestep * norm_mx["load_mw", "sd"] + norm_mx["load_mw", "mean"]
    ) %>%
    na.omit()
  fcst_ts_onestep <- rbind(fcst_ts_onestep, fcst_ts_onestep_add)
}

##### Evaluate one step ahead #####
# RMSE and MAPE by month
fcst_ts_onestep %>%
  group_by(month(Index)) %>%
  summarize(
    rmse_nn = mean(DescTools::RMSE(load_mw, fcst)),
    mape_nn = mean(DescTools::MAPE(load_mw, fcst)),
  )

# Convert to xts
fcsts_xts_onestep <- fcst_ts_onestep %>%
  select(-Index) %>%
  na.omit() %>%
  xts(order.by = na.omit(fcst_ts_onestep)$Index)

# Sample plots
plot(fcsts_xts_onestep["2021-01-01/2021-01-10", c("load_mw", "fcst")])
plot(fcsts_xts_onestep["2021-06-10/2021-06-20", c("load_mw", "fcst")])
plot(fcsts_xts_onestep["2021-09-06/2021-09-15", c("load_mw", "fcst")])
plot(fcsts_xts_onestep["2021-12-20/2021-12-29", c("load_mw", "fcst")])

# Total RMSE and MAPE metrics
DescTools::RMSE(fcsts_xts_onestep$fcst, fcsts_xts_onestep$load_mw)
DescTools::MAPE(fcsts_xts_onestep$fcst, fcsts_xts_onestep$load_mw)

# colnames(fcsts_xts_onestep) <- c("load_mw", "ofic", "nn")
# saveRDS(fcsts_xts_onestep[, -2], "all_forecasts_1_nn.rds")
