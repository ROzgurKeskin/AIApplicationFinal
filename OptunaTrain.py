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

from FinalAnnModel import FinalANN
from library import ProcessDataSet, calculate_head_to_head_win_rate, calculate_recent_points, calculate_win_rate, get_last5_wins
from library import feature_cols, activation_map

# Dataframe'den FTResultEncoded -1 olmayanları ayır
excelDataFrameResult = ProcessDataSet()
df_Ft_model_final= excelDataFrameResult[(excelDataFrameResult['FTResultEncoded'] != -1)]

# Özellikleri güncelle

ftX = df_Ft_model_final[feature_cols].values
ftY = df_Ft_model_final["FTResultEncoded"].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
ftX = scaler.fit_transform(ftX)

import pickle

# Eğitimde scaler'ı fit ettikten sonra kaydediyoruz
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

device = torch.device("cpu")


# Optuna ile ANN Hiperparametre Optimizasyonu
import optuna
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X_train, X_test, y_train, y_test = train_test_split(ftX, ftY, test_size=0.2, random_state=42, stratify=ftY)
# Class weights for imbalance handling
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


def objective(trial):
    # Hiperparametreler
    hyperParameters = {
        "n_hidden1": trial.suggest_int("n_hidden1", 16, 128),  
        "n_hidden2": trial.suggest_int("n_hidden2", 8, 64),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", list(activation_map.keys()))
    }

    model = FinalANN(hyperParameters)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperParameters)


    # Kayıtlı model varsa yükle
    model_path = "model/optuna/best_ann_model_ft.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print("Kayıtlı model yüklendi, eğitime devam ediliyor.")
        except RuntimeError:
            print("Model mimarisi değişti, eski ağırlıklar siliniyor ve eğitim sıfırdan başlatılıyor.")
            os.remove(model_path)

    # Tensör dönüşümü
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Eğitim
    for epoch in range(15):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Test doğruluğu
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        _, predicted = torch.max(preds, 1)
        acc = accuracy_score(y_test_tensor, predicted)
        return acc

# Optuna çalıştırma
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Sonuçlar (en iyi parametreler ve doğruluk)
en_iyi_parametreler = study.best_params
en_iyi_dogruluk = study.best_value
print("En iyi parametreler:", en_iyi_parametreler)
print(f"En iyi doğruluk: {en_iyi_dogruluk * 100:.2f}%")

# --- En iyi modelle yeniden eğitim ve confusion matrix ---

# En iyi parametrelerle tekrar Eğitim
model_final = FinalANN(en_iyi_parametreler)
optimizer = torch.optim.Adam(model_final.parameters(), lr=en_iyi_parametreler['lr'])
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(20):
    model_final.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_final(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# --- Modeli kaydet ---
best_model_path = "model/optuna/best_ann_model_final.pth"
torch.save({
    'model_state_dict': model_final.state_dict(),
    'params': en_iyi_parametreler,
    'feature_cols': feature_cols
}, best_model_path)
print(f"Model kaydedildi: {best_model_path}")

# Kaydedilmiş modeli kullan
checkpoint = torch.load(best_model_path)
loaded_params = checkpoint['params']
loaded_feature_cols = checkpoint['feature_cols']

loaded_model = FinalANN(loaded_params)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()


# Tahmin ve confusion matrix
with torch.no_grad():
    predictions = model_final(X_test_tensor)
    _, predicted_labels = torch.max(predictions, 1)
    acc = accuracy_score(y_test_tensor.cpu(), predicted_labels.cpu())
    print(f"En İyi ANN Model Test Doğruluğu: {acc * 100:.2f}%")
    cm = confusion_matrix(y_test_tensor.cpu(), predicted_labels.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['D', 'H', 'A'])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - En Iyi ANN Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



