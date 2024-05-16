import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import df
import seaborn as sns

print(df["Avg Value"].describe())

# Checking for skew in target variable
plt.hist(df["Avg Value"], bins=30)
plt.title("Avg Value Histogram")
#plt.show()

# Above shows right skew so log transformation applied.
df["Log Avg Value"] = np.log1p(df["Avg Value"])
print(df["Log Avg Value"].describe())
plt.hist(df["Log Avg Value"], bins=30)
plt.title("Log Avg Value Histogram")
#plt.show()

# Looking for patterns/trends with features and the target variable.
n_rows, n_cols = 7, 6
columns = ["Age", "1v1", "Acc", "Aer", "Agg", "Agi", "Ant", "Bal", "Bra", "Cmd", "Cnt", "Cmp", "Cro",
               "Dec", "Det", "Dri", "Fin", "Fir", "Fla", "Han", "Hea", "Jum", "Kic", "Ldr", "Lon", "Mar",
               "OtB", "Pac", "Pas", "Pos", "Ref", "Sta", "Str", "Tck", "Tea", "Tec", "Thr", "TRO", "Vis",
               "Wor", "Cor"]
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,20))
axes = axes.flatten()
for i, col in enumerate(columns):
    axes[i].scatter(df["Log Avg Value"], df[col])
    axes[i].set_title(col)
plt.tight_layout()
#plt.show() #Shows that I need to separate GKs and Outfield players into separate modules


# Checking for correlation between features
corr_matrix = df[["Log Avg Value", "Acc", "Agg", "Agi", "Ant", "Bal", "Bra",  "Cnt", "Cmp", "Cro",
               "Dec", "Det", "Dri", "Fin", "Fir", "Fla",  "Hea", "Jum", "Ldr", "Lon", "Mar",
               "OtB", "Pac", "Pas", "Pos", "Sta", "Str", "Tck", "Tea", "Tec",  "Vis",
               "Wor", "Cor", "Months Remaining", "height_cm", "Wage", "Age"]].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Features")
#plt.show() #Tackling + Marking, and Height + Jumping appear to be highly correlated relative to the other combinations. We will try to merge these features.
