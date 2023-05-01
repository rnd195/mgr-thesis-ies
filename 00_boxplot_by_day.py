## --------------------
## Author: Martin Řanda
## Year: 2023
##
## Python version 3.10.8
##
## Package versions:
## pandas 2.0.0
## matplotlib 3.7.1
##
## --------------------

import pandas as pd
import matplotlib.pyplot as plt

series = pd.read_parquet("data/series_1h_2011_2021.parquet")
series = series.set_index(pd.DatetimeIndex(series["date"]), drop=True)
series = series.drop("date", axis="columns")

series["day"] = [x for x in series.index.weekday]

series.boxplot("load_mw", by="day", figsize=(12, 5), grid=False)
plt.show()
