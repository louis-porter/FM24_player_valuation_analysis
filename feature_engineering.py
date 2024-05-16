import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import df
import seaborn as sns
from EDA import df
from sklearn.preprocessing import StandardScaler

# Removing Goalkeepers from the dataset.
df = df[~df["Position"].str.contains("GK")]


def merge_features(df):
    # Merging features that were flagged in the correlation matrix
    df["Aerial ability"] = (df["height_cm"] + df["Jum"]).mean()
    df["Defensive Ability"] = (df["Tck"] + df["Mar"]).mean()
    return df

def add_categoric_features(df):
    #Capturing top 5 league effect.
    top_5_leagues = ["English Premier Division", "Ligue 1 Uber Eats", "Spanish First Division", "Bundesliga", "Italian Serie A"]
    df["Division"] = df["Division"].str.strip()
    df["top_5_league"] = df["Division"].isin(top_5_leagues).astype(int)
    
    #Capturing "English Tax" effect.
    nationalities = ["ENG"]
    df["Nat"] = df["Nat"].str.strip()
    df["top_10_nations"] = df["Nat"].isin(nationalities).astype(int)

    return df

def scale_features(df):
    #Scales the continiuous vairables
    continuous_features = ["Acc", "Agg", "Agi", "Ant", "Bal", "Bra",  "Cnt", "Cmp", "Cro",
               "Dec", "Det", "Dri", "Fin", "Fir", "Fla",  "Hea", "Jum", "Ldr", "Lon", "Mar",
               "OtB", "Pac", "Pas", "Pos", "Sta", "Str", "Tck", "Tea", "Tec",  "Vis",
               "Wor", "Cor", "Months Remaining", "height_cm", "Wage", "Age", "Aerial ability", "Defensive Ability"]
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    return df



df = merge_features(df)
df= add_categoric_features(df)
scaled_df = scale_features(df)

    


