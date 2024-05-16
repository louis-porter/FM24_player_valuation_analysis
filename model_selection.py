from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from feature_engineering import scaled_df


def train_cv_test_split(df):
    df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
    df_cv, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    return df_train, df_cv, df_test


def split_features(df):
    # Produces one-hot encoding for cateogircal variables
    X_dummies = df[["Acc", "Agg", "Agi", "Ant", "Bal", "Bra",  "Cnt", "Cmp", "Cro",
               "Dec", "Det", "Dri", "Fin", "Fir", "Fla",  "Hea", "Jum", "Ldr", "Lon", "Mar",
               "OtB", "Pac", "Pas", "Pos", "Sta", "Str", "Tck", "Tea", "Tec",  "Vis",
               "Wor", "Cor", "Months Remaining", "height_cm", "Wage", "Age", "Nat", "Division"]]
    X_dummies = pd.get_dummies(X_dummies, columns=["Nat", "Division"], drop_first=True)

    # Produces feature set with merged features from before.
    X_merged = df[["Acc", "Agg", "Agi", "Ant", "Bal", "Bra",  "Cnt", "Cmp", "Cro",
               "Dec", "Det", "Dri", "Fin", "Fir", "Fla",  "Hea", "Ldr", "Lon",
               "OtB", "Pac", "Pas", "Pos", "Sta", "Str", "Tea", "Tec",  "Vis",
               "Wor", "Cor", "Aerial ability", "Defensive Ability",  "Months Remaining", "Wage",
                 "Age", "top_5_league", "top_10_nations"]]
    
    # Produces standard features set
    X_default = df[["Acc", "Agg", "Agi", "Ant", "Bal", "Bra",  "Cnt", "Cmp", "Cro",
               "Dec", "Det", "Dri", "Fin", "Fir", "Fla",  "Hea", "Jum", "Ldr", "Lon", "Mar",
               "OtB", "Pac", "Pas", "Pos", "Sta", "Str", "Tck", "Tea", "Tec",  "Vis",
               "Wor", "Cor", "Months Remaining", "height_cm", "Wage", "Age", "top_5_league", 
               "top_10_nations"]]
    
    y = df["Avg Value"]
    y_log = df["Log Avg Value"]

    return X_dummies, X_merged, X_default, y, y_log


def train_model(x, y):
    # Train linear model and return Mean Squared Error and R-squared score
    model = LinearRegression()
    model.fit(x, y)

    y_pred_log = model.predict(x)
    y_pred = np.exp(y_pred_log)

    mse = mean_squared_error(y, y_pred_log)
    r2 = r2_score(y, y_pred_log)

    mse_original = mean_squared_error(y, y_pred)
    r2_original = r2_score(y, y_pred)

    

    return mse, r2, mse_original, r2_original




df_train, df_cv, df_test = train_cv_test_split(scaled_df)
X_dummies_train, X_merged_train, X_default_train, y_train, y_log_train = split_features(df_train)
X_dummies_cv, X_merged_cv, X_default_cv, y_cv, y_log_cv = split_features(df_cv)
X_dummies_test, X_merged_test, X_default_test, y_test, y_log_test = split_features(df_test)

mse_default_train_log, r2_default_train_log, mse_original, r2_original = train_model(X_default_train, y_log_train)
print(f"MSE: {mse_default_train_log} | RMSE: {np.sqrt(mse_default_train_log)} | R2: {r2_default_train_log}")
print(f"MSE: {mse_original} | RMSE: {np.sqrt(mse_original)} | R2: {r2_original}")


