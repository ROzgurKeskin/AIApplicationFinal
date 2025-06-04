# Optuna ile ANN Hiperparametre Optimizasyonu
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

def objective(trial, feature_cols, X_train, y_train, X_test, y_test):

    # Hiperparametreler
    n_hidden1 = trial.suggest_int("n_hidden1", 16, 128)
    n_hidden2 = trial.suggest_int("n_hidden2", 8, 64)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    activation_name = trial.suggest_categorical("activation", list(activation_map.keys()))

    # Dinamik ANN modeli
    class OptunaANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(len(feature_cols), n_hidden1)
            self.act1 = activation_map[activation_name]
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(n_hidden1, n_hidden2)
            self.act2 = activation_map[activation_name]
            self.dropout2 = nn.Dropout(dropout_rate)
            self.output = nn.Linear(n_hidden2, 3)

        def forward(self, x):
            x = self.dropout1(self.act1(self.fc1(x)))
            x = self.dropout2(self.act2(self.fc2(x)))
            return self.output(x)
        
    
    model = OptunaANN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

def finalOptimization(feature_cols, X_train, y_train, X_test, y_test, best_params):
    # --- En iyi modelle yeniden eğitim ve confusion matrix ---
    class FinalANN(nn.Module):
        def __init__(self, params):
            super().__init__()
            self.fc1 = nn.Linear(len(feature_cols), params['n_hidden1'])
            self.act1 = activation_map[params['activation']]
            self.dropout1 = nn.Dropout(params['dropout'])
            self.fc2 = nn.Linear(params['n_hidden1'], params['n_hidden2'])
            self.act2 = activation_map[params['activation']]
            self.dropout2 = nn.Dropout(params['dropout'])
            self.output = nn.Linear(params['n_hidden2'], 3)

        def forward(self, x):
            x = self.dropout1(self.act1(self.fc1(x)))
            x = self.dropout2(self.act2(self.fc2(x)))
            return self.output(x)

    # Eğitim
    model_final = FinalANN(best_params)
    optimizer = torch.optim.Adam(model_final.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

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

    # Tahmin ve confusion matrix
    model_final.eval()
    with torch.no_grad():
        predictions = model_final(X_test_tensor)
        _, predicted_labels = torch.max(predictions, 1)
        cm = confusion_matrix(y_test_tensor, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix - En Iyi ANN Model")
        plt.show()

# Aktivasyon fonksiyonu eşleme
activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU()
}


