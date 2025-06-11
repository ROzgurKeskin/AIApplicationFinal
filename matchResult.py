import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch

from library import getTeamToIndex
from FinalAnnModel import FinalANN


import torch.nn.functional as F

def predict_match_result(model, scaler, feature_cols, input_dict):
    """
    model: Eğitilmiş PyTorch modeli
    scaler: Eğitimde kullanılan StandardScaler
    feature_cols: Özellik isimleri (sırası önemli)
    input_dict: Kullanıcıdan alınan özellikler (dict)
    """
    # Özellikleri doğru sırayla al
    input_features = [input_dict[col] for col in feature_cols]
    # Numpy array ve ölçekleme
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    # Model ile tahmin
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(output, 1).item())
        predicted_prob = float(probabilities[predicted_class])
    # Sonucu etikete çevir
    result_map = {0: "Beraberlik (D)", 1: "Ev Sahibi Kazanır (H)", 2: "Deplasman Kazanır (A)"}
    return result_map.get(predicted_class, "Bilinmiyor"), predicted_prob

def predict_with_team_names(model, scaler, feature_cols, team_to_index, input_teams):
    """
    input_teams: {'FirstTeam': 'Besiktas', 'SecondTeam': 'Adana'}
    """
    # Takım isimlerini encode et
    first_team_encoded = team_to_index.get(input_teams['FirstTeam'], 0)
    second_team_encoded = team_to_index.get(input_teams['SecondTeam'], 0)
    # Diğer özellikleri sıfırla veya uygun default değer ata
    input_dict = {
        "FirstTeamEncoded": first_team_encoded,
        "SecondTeamEncoded": second_team_encoded,
    }
    # Diğer feature'ları sıfırla
    for col in feature_cols:
        if col not in input_dict:
            input_dict[col] = 0
    # Tahmin fonksiyonunu çağır
    return predict_match_result(model, scaler, feature_cols, input_dict)

# Kaydedilmiş modeli kullan
best_model_path = "model/optuna/best_ann_model_final.pth"
checkpoint = torch.load(best_model_path)
loaded_params = checkpoint['params']
loaded_feature_cols = checkpoint['feature_cols']

loaded_model = FinalANN(loaded_params)


import pickle

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

team_to_index = getTeamToIndex()

input_teams = {'FirstTeam': 'Antalya', 'SecondTeam': 'Kayseri'}
tahmin, olasilik = predict_with_team_names(loaded_model, scaler, loaded_feature_cols, team_to_index, input_teams)
print(f"Tahmin edilen sonuç: {tahmin} (Olasılık:% {olasilik*100:.2f})")