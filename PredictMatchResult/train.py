# PyTorch'u CPU ile çalıştıracak şekilde yeniden yüklüyoruz (GPU zorunlu olmayan mod)
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from annModel import ANNModel

import sys
print(sys.getdefaultencoding())
# -*- coding: utf-8 -*-

# Excel dosyasını tekrar oku
matchResultSourceFile = "C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults.xlsx"
df = pd.read_excel(matchResultSourceFile)

# Takım isimlerini encode et
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

# Son 5 maç galibiyet sayısı ve ev sahibi avantajı ekle
def get_last5_wins(team, date, df, team_col, result_col):
    matches = df[(df[team_col] == team) & (df['MatchDate'] < date)].sort_values('MatchDate', ascending=False).head(5)
    if team_col == 'FirstTeam':
        return (matches[result_col] == 1).sum()
    else:
        return (matches[result_col] == 2).sum()

df['FirstTeamLast5Wins'] = df.apply(lambda row: get_last5_wins(row['FirstTeam'], row['MatchDate'], df, 'FirstTeam', 'FTResultEncoded'), axis=1)
df['SecondTeamLast5Wins'] = df.apply(lambda row: get_last5_wins(row['SecondTeam'], row['MatchDate'], df, 'SecondTeam', 'FTResultEncoded'), axis=1)
df['FirstTeamLast5HTWins'] = df.apply(lambda row: get_last5_wins(row['FirstTeam'], row['MatchDate'], df, 'FirstTeam', 'HTResultEncoded'), axis=1)
df['SecondTeamLast5HTWins'] = df.apply(lambda row: get_last5_wins(row['SecondTeam'], row['MatchDate'], df, 'SecondTeam', 'HTResultEncoded'), axis=1)
df['IsHome'] = 1  # Ev sahibi için 1, deplasman için 0 (örnek olarak tümü ev sahibi)

# Dataframe'den FTResultEncoded -1 olmayanları ayır
df_Ft_model_final = df[(df['FTResultEncoded'] != -1)|(df['HTResultEncoded'] != -1)]

# Özellikleri güncelle
feature_cols = ["FirstTeamEncoded", "SecondTeamEncoded", "FirstTeamLast5Wins", "SecondTeamLast5Wins","FirstTeamLast5HTWins", "SecondTeamLast5HTWins", "IsHome"]
ftX = df_Ft_model_final[feature_cols].values
ftY = df_Ft_model_final["FTResultEncoded"].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
ftX = scaler.fit_transform(ftX)

# Encoding işlemlerinden sonra veriyi kaydet
df.to_excel("C:\\PythonProject\\Github Project\\YapayZekaFinal\\Sample_AllMatchsResults_encoded.xlsx", index=False)

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(ftX, ftY, test_size=0.2, random_state=42, stratify=ftY)

# CPU uyumlu tensörler
device = torch.device("cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, kayıp ve optimizer
model = ANNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Kayıtlı model varsa yükle
model_path = "model/trained_ann_model.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        print("Kayıtlı model yüklendi, eğitime devam ediliyor.")
    except RuntimeError:
        print("Model mimarisi değişti, eski ağırlıklar siliniyor ve eğitim sıfırdan başlatılıyor.")
        os.remove(model_path)

# Eğitim
best_loss = float('inf')
patience = 10  # Kaç epoch boyunca iyileşme olmazsa duracak
trigger_times = 0

num_epochs = 50  # Daha uzun eğitim için artırabilirsiniz
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Doğrulama kaybı hesapla
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

    print(f"Epoch {epoch+1}, Validation Loss: {val_loss.item():.4f}")

    # Early stopping kontrolü
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        trigger_times = 0
        # En iyi modeli kaydet
        torch.save(model.state_dict(), "model/best_ann_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping: Eğitim durduruldu.")
            break

# Eğitim sonrası en iyi modeli yükle
model.load_state_dict(torch.load("model/best_ann_model.pth"))

# FTResultEncoded için test doğruluğu
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy_ft = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
print(f"Full Time (FTResultEncoded) Test Accuracy: {accuracy_ft * 100:.2f}%")

# HTResultEncoded için test doğruluğu
# Önce HTResultEncoded değerlerini ve uygun X_test'i hazırlayın
htY = df_Ft_model_final["HTResultEncoded"].values
_, X_test_ht, _, y_test_ht = train_test_split(ftX, htY, test_size=0.2, random_state=42, stratify=htY)
X_test_ht_tensor = torch.tensor(X_test_ht, dtype=torch.float32).to(device)
y_test_ht_tensor = torch.tensor(y_test_ht, dtype=torch.long).to(device)

with torch.no_grad():
    outputs_ht = model(X_test_ht_tensor)
    _, predicted_ht = torch.max(outputs_ht, 1)
    accuracy_ht = accuracy_score(y_test_ht_tensor.cpu(), predicted_ht.cpu())
print(f"Half Time (HTResultEncoded) Test Accuracy: {accuracy_ht * 100:.2f}%")
