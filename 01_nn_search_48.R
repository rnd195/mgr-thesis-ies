## --------------------
## Description: Neural network models grid search 48h ahead
## Author: Martin Řanda
## Year: 2023
##
## R Version 4.2.2
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
librarian::shelf(tsibble, xts, dplyr, rstudioapi, tidyverse, torch, lubridate, DescTools)
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
# 1h load, weather, price and forecasts
series <- readRDS("data/series_features_1h_2012_2021_v2.rds")
dummies <- readRDS("data/seasonal_dummies_3.rds")

all_data <- merge(series["2012/"], dummies["2012/"])
head(all_data)

#### Prepare data ####
# Convert data to tsibble format
all_data_ts <- all_data %>%
  fortify.zoo() %>%
  as_tsibble()

all_data_ts_2021 <- all_data_ts %>%
  filter(year(Index) == 2021) %>%
  select(load_mw, Index)


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
train_set <- c("2017-01-01", "2020-05-31")
valid_set <- c("2020-06-01", "2020-12-31")
test_set <- c("2021-01-01", "2021-12-31")

# Be careful with changing this parameter due to the intersection with the oos set
# -> only multiples of 24
n_timesteps <- 5 * 24
# 48 hours ahead
n_forecast <- 2 * 24
# Hyperparameter in the optimizer
batch_size <- 128

# Change the start of the test_set so that, after the lookback parameter (n_timesteps)
# is considered, the predictions start at 2021-01-01 00:00
test_set[1] <- as.Date(test_set[1]) %m-% days(n_timesteps / 24) %>%
  as.character()

# Generate in-sample (train), validation, and out-of-sample (test) data
train_data <- subset_data(train_set[1], train_set[2], vars = variables) %>% as.matrix()
valid_data <- subset_data(valid_set[1], valid_set[2], vars = variables) %>% as.matrix()
test_data <- subset_data(test_set[1], test_set[2], vars = variables) %>% as.matrix()

# Save mean and sd for normalization and denormalization in a matrix
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

# Define hyperparameter grid, batch size is defined earlier
hyper_grid_tmp <- expand.grid(
  hidden_size = c(64, 128, 192),
  linear_size = c(128, 256, 512),
  lstm_layers = c(1, 2),
  dropout = c(0, 0.2),
  linear_dropout = c(0, 0.2),
  learning_rate = c(0.001, 0.0005),
  epochs = 30
)
# Drop dropouts for lstm_layers = 1 (no droupout necessary)
hyper_grid <- hyper_grid_tmp[!with(hyper_grid_tmp, lstm_layers == 1 & dropout > 0), ]
rownames(hyper_grid) <- 1:nrow(hyper_grid)

# Training and validation loss
hyper_grid[, paste0("tl_e", 1:30)] <- rep(NA, nrow(hyper_grid))
hyper_grid[, paste0("vl_e", 1:30)] <- rep(NA, nrow(hyper_grid))

#### Run the 48h grid search ####

for (j in 1:nrow(hyper_grid)) {
  # Set LSTM parameters
  net <- model_lstm(
    input_size = length(variables),
    hidden_size = hyper_grid$hidden_size[j],
    linear_size = hyper_grid$linear_size[j],
    num_layers = hyper_grid$lstm_layers[j],
    output_size = n_forecast,
    dropout = hyper_grid$dropout[j],
    linear_dropout = hyper_grid$linear_dropout[j]
  )

  # Fit
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  net <- net$to(device = device)

  # Specify the number of epochs and the optimizer
  num_epochs <- hyper_grid$epochs[j]
  optimizer <- optim_adam(net$parameters, lr = hyper_grid$learning_rate[j])

  # Perform training
  valid_losses <- rep(NA, num_epochs)
  train_losses <- rep(NA, num_epochs)
  cat("\nIteration", j, "\n")
  for (epoch in 1:num_epochs) {
    net$train()
    train_loss <- c()
    coro::loop(for (b in train_dl) {
      loss <- train_batch(b)
      train_loss <- c(train_loss, loss)
    })
    cat(sprintf("\nEpoch %d, training loss: %3.5f", epoch, mean(train_loss)))
    train_losses[epoch] <- mean(train_loss)

    net$eval()
    valid_loss <- c()
    coro::loop(for (b in valid_dl) {
      loss <- valid_batch(b)
      valid_loss <- c(valid_loss, loss)
    })
    cat(sprintf("\nEpoch %d, validation loss: %3.5f\n", epoch, mean(valid_loss)))
    valid_losses[epoch] <- mean(valid_loss)
  }
  cat("\n---------------------------------\n")

  hyper_grid[j, paste0("tl_e", 1:30)] <- train_losses
  hyper_grid[j, paste0("vl_e", 1:30)] <- valid_losses

  cat(j, "/", nrow(hyper_grid), "\n")
  write.csv(na.omit(hyper_grid), "nn_hyper_grid_48h.csv")
  # Clean up
  rm(net)
}
