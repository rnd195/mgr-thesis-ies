## --------------------
## Description: Initial variable selection for the neural network models
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
## rpart 4.1.19
## ipred 0.9.13
## caret 6.0.93
## readxl 1.4.2
## ggplot2 3.4.1
## librarian 1.8.1
##
## --------------------

if (!require(librarian)) install.packages("librarian")
librarian::shelf(dplyr, xts, DescTools, rstudioapi, rpart, ipred, caret, readxl, ggplot2)
# Set timezone to CET, locale to English, and working directory to filepath
Sys.setenv(TZ = "CET")
Sys.setlocale("LC_TIME", "English")
setwd(dirname(getActiveDocumentContext()$path))
set.seed(1111)

# Load, weather, price, official forecasts, and constructed features
series <- readRDS("data/series_features_1h_2012_2021_v2.rds")["2012/2020"]
# Constructed dummy variables
dummies <- readRDS("data/seasonal_dummies_3.rds")["2012/2020"]

df <- as.data.frame(cbind(series, dummies)) %>%
  select(-load_mw_f)

# Run bagged regression tree with 30 replications
tree <- bagging(
  load_mw ~ .,
  data = df,
  nbagg = 30,
  control = rpart.control(
    maxdepth = 18,
    minsplit = 21,
    cp = 5.05e-07
  )
)

# RMSE and variable importance
cat(DescTools::RMSE(predict(tree, df), df$load_mw), "in sample RMSE")
# 80.63676 in sample RMSE

important_vars <- varImp(tree) %>%
  arrange(desc(Overall))

# To relative importance with respect to the top performer
top_var <- as.numeric(head(important_vars, 1))
important_vars$Relative <- as.numeric(important_vars$Overall / top_var)
important_vars

# Export table
write.csv(important_vars, "important_vars.csv")

# Variable importance barplot
# important_vars <- read_excel("important_vars.xlsx")

important_vars %>%
  head(15) %>%
  ggplot(aes(x = Relative, y = reorder(`Variable (2012/2020) data`, Relative))) +
  geom_bar(stat = "identity") +
  theme_bw()

