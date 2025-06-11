# PyTorch'u CPU ile çalıştıracak şekilde yeniden yüklüyoruz (GPU zorunlu olmayan mod)
import os
from sklearn.utils import compute_class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from annModel import ANNModel
from library import ProcessDataSet, calculate_head_to_head_win_rate, calculate_recent_points, calculate_win_rate, get_last5_wins
from library import feature_cols

# Dataframe'den FTResultEncoded -1 olmayanları ayır
excelDataFrameResult = ProcessDataSet()
df_Ft_model_final= excelDataFrameResult[(excelDataFrameResult['FTResultEncoded'] != -1)]


ftX = df_Ft_model_final[feature_cols].values
ftY = df_Ft_model_final["FTResultEncoded"].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
ftX = scaler.fit_transform(ftX)

device = torch.device("cpu")


# --- FTResultEncoded için eğitim ve test ---
print("\n--- Full Time (FTResultEncoded) Model Eğitimi ---")
X_train, X_test, y_train, y_test = train_test_split(ftX, ftY, test_size=0.2, random_state=42, stratify=ftY)

# Class weights for imbalance handling
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model_ft = ANNModel().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model_ft.parameters(), lr=0.0005)

# Kayıtlı model varsa yükle
model_path = "model/best_ann_model_ft.pth"
if os.path.exists(model_path):
    try:
        model_ft.load_state_dict(torch.load(model_path))
        print("Kayıtlı model yüklendi, eğitime devam ediliyor.")
    except RuntimeError:
        print("Model mimarisi değişti, eski ağırlıklar siliniyor ve eğitim sıfırdan başlatılıyor.")
        os.remove(model_path)


best_loss = float('inf')
patience = 10
trigger_times = 0
num_epochs = 10

for epoch in range(num_epochs):
    model_ft.train()
    for inputs, labels in train_loader:
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model_ft.eval()
    with torch.no_grad():
        val_outputs = model_ft(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    print(f"[FT] Epoch {epoch+1}, Validation Loss: {val_loss.item():.4f}")
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        trigger_times = 0
        torch.save(model_ft.state_dict(), "model/best_ann_model_ft.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("[FT] Early stopping: Eğitim durduruldu.")
            break

model_ft.load_state_dict(torch.load("model/best_ann_model_ft.pth"))
model_ft.eval()
with torch.no_grad():
    outputs = model_ft(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy_ft = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
print(f"Full Time (FTResultEncoded) Test Accuracy: {accuracy_ft * 100:.2f}%")



# --- HTResultEncoded için eğitim ve test ---
print("\n--- Half Time (HTResultEncoded) Model Eğitimi ---")


df_Ht_model_final = excelDataFrameResult[(excelDataFrameResult['HTResultEncoded'] != -1)]

htY = df_Ht_model_final["HTResultEncoded"].values
htX = df_Ht_model_final[feature_cols].values

scaler = StandardScaler()
htX = scaler.fit_transform(htX)

X_train_ht, X_test_ht, y_train_ht, y_test_ht = train_test_split(htX, htY, test_size=0.2, random_state=42, stratify=htY)
X_train_ht_tensor = torch.tensor(X_train_ht, dtype=torch.float32).to(device)
y_train_ht_tensor = torch.tensor(y_train_ht, dtype=torch.long).to(device)
X_test_ht_tensor = torch.tensor(X_test_ht, dtype=torch.float32).to(device)
y_test_ht_tensor = torch.tensor(y_test_ht, dtype=torch.long).to(device)

train_dataset_ht = TensorDataset(X_train_ht_tensor, y_train_ht_tensor)
train_loader_ht = DataLoader(train_dataset_ht, batch_size=32, shuffle=True)

model_ht = ANNModel().to(device)
criterion_ht = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_ht = optim.Adam(model_ht.parameters(), lr=0.0005)

# Kayıtlı model varsa yükle
model_path_ht = "model/best_ann_model_ht.pth"
if os.path.exists(model_path_ht):
    try:
        model_ht.load_state_dict(torch.load(model_path_ht))
        print("Kayıtlı model yüklendi, eğitime devam ediliyor.")
    except RuntimeError:
        print("Model mimarisi değişti, eski ağırlıklar siliniyor ve eğitim sıfırdan başlatılıyor.")
        os.remove(model_path_ht)


best_loss_ht = float('inf')
trigger_times_ht = 0

for epoch in range(num_epochs):
    model_ht.train()
    for inputs, labels in train_loader_ht:
        outputs = model_ht(inputs)
        loss = criterion_ht(outputs, labels)
        optimizer_ht.zero_grad()
        loss.backward()
        optimizer_ht.step()
    model_ht.eval()
    with torch.no_grad():
        val_outputs = model_ht(X_test_ht_tensor)
        val_loss = criterion_ht(val_outputs, y_test_ht_tensor)
    print(f"[HT] Epoch {epoch+1}, Validation Loss: {val_loss.item():.4f}")
    if val_loss.item() < best_loss_ht:
        best_loss_ht = val_loss.item()
        trigger_times_ht = 0
        torch.save(model_ht.state_dict(), "model/best_ann_model_ht.pth")
    else:
        trigger_times_ht += 1
        if trigger_times_ht >= patience:
            print("[HT] Early stopping: Eğitim durduruldu.")
            break

model_ht.load_state_dict(torch.load("model/best_ann_model_ht.pth"))
model_ht.eval()
with torch.no_grad():
    outputs_ht = model_ht(X_test_ht_tensor)
    _, predicted_ht = torch.max(outputs_ht, 1)
    accuracy_ht = accuracy_score(y_test_ht_tensor.cpu(), predicted_ht.cpu())
print(f"Half Time (HTResultEncoded) Test Accuracy: {accuracy_ht * 100:.2f}%")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# FTResultEncoded için confusion matrix
cm_ft = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
plt.figure(figsize=(5,4))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues', xticklabels=['D', 'H', 'A'], yticklabels=['D', 'H', 'A'])
plt.title('Confusion Matrix - Full Time (FTResultEncoded)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# HTResultEncoded için confusion matrix
cm_ht = confusion_matrix(y_test_ht_tensor.cpu(), predicted_ht.cpu())
plt.figure(figsize=(5,4))
sns.heatmap(cm_ht, annot=True, fmt='d', cmap='Blues', xticklabels=['D', 'H', 'A'], yticklabels=['D', 'H', 'A'])
plt.title('Confusion Matrix - Half Time (HTResultEncoded)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()