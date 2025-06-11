import os
import pickle

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch.nn as nn

feature_cols = ["FirstTeamEncoded", "SecondTeamEncoded", 
                "FirstTeamLast5Wins", "SecondTeamLast5Wins",
                "FirstTeamLast5HTWins", "SecondTeamLast5HTWins",
                "FirstTeamWinRate", "SecondTeamWinRate",
                "FirstTeamRecentPoints", "SecondTeamRecentPoints",
                "HeadToHeadWinRate"]

# Aktivasyon fonksiyonu eşleme
activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU()
}


def calculate_win_rate(team, current_date, df, team_col, result_col, win_value):
    past_matches = df[(df[team_col] == team) & (df["MatchDate"] < current_date)]
    if len(past_matches) == 0:
        return 0.0
    wins = (past_matches[result_col] == win_value).sum()
    return wins / len(past_matches)

# Son 5 maç galibiyet sayısı ve ev sahibi avantajı ekle
def get_last5_wins(team, date, df, team_col, result_col):
    matches = df[(df[team_col] == team) & (df['MatchDate'] < date)].sort_values('MatchDate', ascending=False).head(5)
    if team_col == 'FirstTeam':
        return (matches[result_col] == 1).sum()
    else:
        return (matches[result_col] == 2).sum()

def calculate_recent_points(team, current_date, df, team_col, result_col, win_val, draw_val):
    matches = df[(df[team_col] == team) & (df["MatchDate"] < current_date)].sort_values("MatchDate", ascending=False).head(3)
    points = 0
    for r in matches[result_col]:
        if r == win_val:
            points += 3
        elif r == draw_val:
            points += 1
    return points

def calculate_head_to_head_win_rate(home, away, date, df):
    past_matches = df[((df["FirstTeam"] == home) & (df["SecondTeam"] == away)) |
                      ((df["FirstTeam"] == away) & (df["SecondTeam"] == home))]
    past_matches = past_matches[past_matches["MatchDate"] < date]
    if past_matches.empty:
        return 0.5  # nötr değer
    home_wins = ((past_matches["FirstTeam"] == home) & (past_matches["FTResultEncoded"] == 1)).sum()
    away_wins = ((past_matches["SecondTeam"] == home) & (past_matches["FTResultEncoded"] == 2)).sum()
    total = len(past_matches)
    return (home_wins + away_wins) / total

def getTeamToIndex():
    matchResultEncodedFile = "C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults_encoded.xlsx"
    if os.path.exists(matchResultEncodedFile):
        df = pd.read_excel(matchResultEncodedFile)
        team_to_index = {team: idx for idx, team in enumerate(pd.unique(df[['FirstTeam', 'SecondTeam']].values.ravel()))}
        return team_to_index
    else:
        print("Encoded dosyası bulunamadı.")
        return {}


def ProcessDataSet():
    # Excel dosyasını tekrar oku
    matchResultSourceFile = "C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults.xlsx"
    matchResultEncodedFile = "C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults_encoded.xlsx"

    # Eğer encoded dosya yoksa, veriyi encode et ve kaydet
    if os.path.exists(matchResultEncodedFile)==False:
        print("Encoded dosyası yok, datalar encode ediliyor...")
        df = pd.read_excel(matchResultSourceFile)
        team_to_index = {team: idx for idx, team in enumerate(pd.unique(df[['FirstTeam', 'SecondTeam']].values.ravel()))}
        df['FirstTeamEncoded'] = df['FirstTeam'].map(team_to_index)
        df['SecondTeamEncoded'] = df['SecondTeam'].map(team_to_index)

        # Sonuçları sayısallaştır
        def encode_result(result):
            if result == 'H':
                return 1
            elif result == 'D':
                return 0
            elif result == 'A':
                return 2
            else:
                return -1

        df['HTResultEncoded'] = df['HTResultCode'].apply(encode_result)
        df['FTResultEncoded'] = df['FTResultCode'].apply(encode_result)
        df['FirstTeamLast5Wins'] = df.apply(lambda row: get_last5_wins(row['FirstTeam'], row['MatchDate'], df, 'FirstTeam', 'FTResultEncoded'), axis=1)
        df['SecondTeamLast5Wins'] = df.apply(lambda row: get_last5_wins(row['SecondTeam'], row['MatchDate'], df, 'SecondTeam', 'FTResultEncoded'), axis=1)
        df['FirstTeamLast5HTWins'] = df.apply(lambda row: get_last5_wins(row['FirstTeam'], row['MatchDate'], df, 'FirstTeam', 'HTResultEncoded'), axis=1)
        df['SecondTeamLast5HTWins'] = df.apply(lambda row: get_last5_wins(row['SecondTeam'], row['MatchDate'], df, 'SecondTeam', 'HTResultEncoded'), axis=1)
        df["FirstTeamWinRate"] = df.apply(lambda row: calculate_win_rate(row["FirstTeam"], row["MatchDate"], df, "FirstTeam", "FTResultEncoded", 1), axis=1)
        df["SecondTeamWinRate"] = df.apply(lambda row: calculate_win_rate(row["SecondTeam"], row["MatchDate"], df, "SecondTeam", "FTResultEncoded", 2), axis=1)
        df["FirstTeamRecentPoints"] = df.apply(lambda row: calculate_recent_points(row["FirstTeam"], row["MatchDate"], df, "FirstTeam", "FTResultEncoded", 1, 0), axis=1)
        df["SecondTeamRecentPoints"] = df.apply(lambda row: calculate_recent_points(row["SecondTeam"], row["MatchDate"], df, "SecondTeam", "FTResultEncoded", 2, 0), axis=1)
        df["HeadToHeadWinRate"] = df.apply(lambda row: calculate_head_to_head_win_rate(row["FirstTeam"], row["SecondTeam"], row["MatchDate"], df), axis=1)
        # Takım isimlerini encode et

        # df['FirstTeamLast5Wins'] = 2
        # df['SecondTeamLast5Wins'] = 3  
        # df['FirstTeamLast5HTWins'] = 4
        # df['SecondTeamLast5HTWins'] = 5
        # df['IsHome'] = 1  # Ev sahibi için 1, deplasman için 0 (örnek olarak tümü ev sahibi)
        # Encoding işlemlerinden sonra veriyi kaydet
        df.to_excel("C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults_encoded.xlsx", index=False)
    else:
        df = pd.read_excel(matchResultEncodedFile)
    return df

def getScaler():
    scaler_file = "model/scaler.pkl"
    if os.path.exists(scaler_file):
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        return scaler, True
    else:
        print("Scaler dosyası bulunamadı. Yeni bir scaler oluşturuluyor.")
        return StandardScaler(), False






