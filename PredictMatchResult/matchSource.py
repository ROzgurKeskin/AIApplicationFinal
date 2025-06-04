import pandas as pd

matchResultSourceFile = "C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults.xlsx"

# Dosyayı oku
df = pd.read_excel(matchResultSourceFile, sheet_name='Sayfa1')

# İlk 5 satırı görüntüle
print(df.head())

# Temel bilgi
print(df.info())

from sklearn.preprocessing import LabelEncoder

# Kullanılacak sütunlar
df_model = df[["FirstTeam", "SecondTeam", "FTResultCode"]].copy()

# Takım isimlerini sayısal değerlere dönüştür
team_encoder = LabelEncoder()
df_model["FirstTeamEncoded"] = team_encoder.fit_transform(df_model["FirstTeam"])
df_model["SecondTeamEncoded"] = team_encoder.transform(df_model["SecondTeam"])

# Hedef değişkeni (Sonuç)
result_encoder = LabelEncoder()
df_model["ResultEncoded"] = result_encoder.fit_transform(df_model["FTResultCode"])

# Modelde kullanılacak son veri
df_model_final = df_model[["FirstTeamEncoded", "SecondTeamEncoded", "ResultEncoded"]]

df_model_final.head()