# PyTorch'u CPU ile çalıştıracak şekilde yeniden yüklüyoruz (GPU zorunlu olmayan mod)
import os
from sklearn.utils import compute_class_weight
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from FinalAnnModel import FinalANN
from library import ProcessDataSet
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


# Class weights for imbalance handling
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(ftY), y=ftY)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


def objective(trial):
    # Hiperparametreler
    # Hiperparametreleri json objesine çeviriyoruz
    # Optuna'nın deneme fonksiyonunda kullanılacak hiperparametreler
    hyperParameters = {
        "n_hidden1": trial.suggest_int("n_hidden1", 16, 128),  
        "n_hidden2": trial.suggest_int("n_hidden2", 8, 64),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", list(activation_map.keys()))
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(ftX, ftY)):
        X_train, X_val = ftX[train_index], ftX[val_index]
        y_train, y_val = ftY[train_index], ftY[val_index]

        model = FinalANN(hyperParameters)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperParameters['lr'])

        # Tensör dönüşümü
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Eğitim
        for epoch in range(15): # Her katlama için eğitim epokları
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Doğrulama doğruluğu
        model.eval()
        with torch.no_grad():
            preds = model(X_val_tensor)
            _, predicted = torch.max(preds, 1)
            acc = accuracy_score(y_val_tensor, predicted)
            fold_accuracies.append(acc)
    
    return np.mean(fold_accuracies) # Ortalama doğruluğu döndür

# Optuna çalıştırma
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5) # Deneme sayısını artırabiliriz

# Sonuçlar (en iyi parametreler ve doğruluk)
en_iyi_parametreler = study.best_params
en_iyi_dogruluk = study.best_value
print("En iyi parametreler:", en_iyi_parametreler)
print(f"En iyi doğruluk (Ortalama Çapraz Doğrulama): {en_iyi_dogruluk * 100:.2f}%")


# En iyi parametrelerle tüm veri üzerinde tekrar Eğitim
model_final = FinalANN(en_iyi_parametreler)
optimizer = torch.optim.Adam(model_final.parameters(), lr=en_iyi_parametreler['lr'])
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

X_tensor_full = torch.tensor(ftX, dtype=torch.float32)
y_tensor_full = torch.tensor(ftY, dtype=torch.long)

full_dataset = TensorDataset(X_tensor_full, y_tensor_full)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

for epoch in range(20): # Tüm veri üzerinde eğitim epokları
    model_final.train()
    for inputs, labels in full_loader:
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

# Tahmin ve confusion matrix için yeni bir test seti oluştur (veya mevcut test setini kullan)
# Burada tüm veri üzerinde eğitim yapıldığı için, gerçek bir test seti ayırmak yerine
# çapraz doğrulama ile zaten değerlendirme yapıldı. Ancak yine de bir örnek göstermek için
# tüm veri üzerindeki performans gösterilebilir.
# Gerçek dünyada modelin daha önce görmediği verilere nasıl tepki verdiğini görmek için
# orijinal train_test_split yapısını koruyup son değerlendirmeyi ayrı bir test setinde yapmak daha iyi olur.

# Mevcut ftX ve ftY'yi test olarak kullanabiliriz veya yeniden train_test_split yapabiliriz.
# Çapraz doğrulama yapıldığı için bu adım genellikle gereksizdir.
# Ancak modelin genel performansını tüm veri üzerinde görmek için yapılabilir.
with torch.no_grad():
    predictions = loaded_model(X_tensor_full) # Eğitim yapılan tüm veri üzerinde tahmin
    _, predicted_labels = torch.max(predictions, 1)
    acc = accuracy_score(y_tensor_full.cpu(), predicted_labels.cpu())
    print(f"Eğitilmiş ANN Modelin Tüm Veri Üzerindeki Doğruluğu: {acc * 100:.2f}%")
    cm = confusion_matrix(y_tensor_full.cpu(), predicted_labels.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['D', 'H', 'A'])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Eğitilmiş ANN Model (Tüm Veri Üzerinde)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()