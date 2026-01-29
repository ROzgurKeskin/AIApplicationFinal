
# Hugging Face Inference Providers API ile undefined inputları modern şekilde sorgulama
from huggingface_hub import InferenceClient

def hf_inference_undefined_input(text, api_key, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Hugging Face Inference Providers API ile verilen inputu chat completion formatında sınıflandırır.
    Args:
        text (str): Sorgulanacak metin
        api_key (str): Hugging Face API anahtarı
        model (str): Kullanılacak model adı (chat completion destekleyen bir model olmalı)
    Returns:
        str: Modelin döndürdüğü yanıt
    """
    categories = [
        "Bağlantı sorunu", "Donanım talebi", "Erişim hatası", "Erişim talebi",
        "Hata mesajı alıyorum", "Mail grubu ekleme", "Rapor talebi", "Sistem hatası",
        "Sunucuya ulaşılamıyor", "Uygulama çalışmıyor", "VPN bağlantı problemi",
        "VPN erişim isteği", "Yazılım kurulumu", "Yeni kullanıcı isteği", "Yetki talebi"
    ]
    prompt = f"Sen bir IT Destek Uzmanısın. Sadece listeden EN UYGUN kategoriyi seç, açıklama yapma sadece seçtiğin kategoriyi cevap olarak yaz sohbet modunu kapat: {', '.join(categories)}\n\nMesaj: {text}"
    client = InferenceClient(token=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# Örnek kullanım:
# api_key = "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"
# result = hf_inference_undefined_input("Buraya undefined input yazılır", api_key)
# print(result)


import re
import json
import os
import pandas as pd

def kategorize_metni(metin):
    # Talep için örnek regex (istek, rica, lütfen, yapabilir misin vb.)
    talep_regex = r"(istek|rica|lütfen|yapabilir misin|yapar mısın|gönder|ekle|oluştur|başlat|aç|kapat|getir|göster|kurum|güncelle|sil|düzenle)"
    # Hata için örnek regex (hata, çalışmıyor, sorun, uyarı, exception, error vb.)
    hata_regex = r"(hata|çalışmıyor|sorun|uyarı|exception|error|yanlış|fail|bug|ulaşılamıyor|çökmüş|problem|donuyor|kilitleniyor|atıyor)"

    if re.search(hata_regex, metin, re.IGNORECASE):
        return "hata"
    elif re.search(talep_regex, metin, re.IGNORECASE):
        return "talep"
    else:
        chatResponse= hf_inference_undefined_input(metin, api_key="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")
        if chatResponse=="":
            return "Tanımsız"
        return chatResponse


def temizle_metin(metin):
    # Tüm kaçış karakterlerini ve \n, \r, \t, \f, \v gibi karakterleri temizle
    temiz = re.sub(r'(\\n|\\r|\\t|\\f|\\v|\n|\r|\t|\f|\v)+', ' ', metin)
    temiz = re.sub(r' +', ' ', temiz)
    return temiz.strip()

def toplu_kategorize_ve_ekle_body(excel_path, intents_path):
    df = pd.read_excel(excel_path, sheet_name=0)
    if 'body' not in df.columns:
        print("1. sayfada 'body' isimli bir kolon bulunamadı.")
        return
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    eklendi = 0
    for metin in df['body'].dropna().astype(str):
        temiz_metin = temizle_metin(metin)
        sonuc = kategorize_metni(temiz_metin)
        if sonuc == "hata":
            kategori = "Hata"
        elif sonuc == "talep":
            kategori = "Talep"
        else:
            kategori = "Undefined"
            
        for intent in data["intents"]:
            if intent["MainCategory"].lower() == kategori.lower():
                if temiz_metin not in intent["patterns"]:
                    intent["patterns"].append(temiz_metin)
                    eklendi += 1
                break
    with open(intents_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, separators=(",", ": "))
    print(f"Toplam {eklendi} kayıt eklendi.")

if __name__ == "__main__":
    intents_path = os.path.join(os.path.dirname(__file__), "intents.json")
    undefined_input_path = os.path.join(os.path.dirname(__file__), "undefined_input.json")
    excel_path = os.path.join(os.path.dirname(__file__), "IT_Department_Sorunlar_.xlsx")
    while True:
        print("\n1- Tek tek metin gir\n2- Excel'den toplu işle\n3- Çıkış")
        secim = input("Seçiminiz: ")
        if secim == "1":
            metin = input("Metni girin (çıkmak için 'exit' yazın): ")
            if metin.strip().lower() == "exit":
                print("Program sonlandırıldı.")
                break
            temiz_metin = temizle_metin(metin)
            sonuc = kategorize_metni(temiz_metin)
            print(f'Girilen metin: "{temiz_metin}" -> Kategori: {sonuc}')
            with open(intents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(undefined_input_path, "r", encoding="utf-8") as f:
                undefined_data = json.load(f)
            
            if sonuc == "hata":
                kategori = "Hata"
            elif sonuc == "talep":
                kategori = "Talep"
            else:
                kategori = "Undefined"
            if  kategori=="Undefined":
                # undefined_input.json dosyasında 'undefined_inputs' yok, 'patterns' anahtarını kullan
                for intent in undefined_data.get("intents", []):
                    if intent.get("MainCategory", "").lower() == "undefined":
                        if temiz_metin not in intent["patterns"]:
                            intent["patterns"].append(temiz_metin)
                        break
                with open(undefined_input_path, "w", encoding="utf-8") as f:
                    json.dump(undefined_data, f, ensure_ascii=False, indent=2, separators=(",", ": "))
                print("Kategori 'Undefined' olduğu için undefined_input.json dosyasına kaydedildi.")
                continue

            for intent in data["intents"]:
                if intent["MainCategory"].lower() == kategori.lower():
                    if temiz_metin not in intent["patterns"]:
                        intent["patterns"].append(temiz_metin)
                    break
            with open(intents_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, separators=(",", ": "))
        elif secim == "2":
            toplu_kategorize_ve_ekle_body(excel_path, intents_path)
        elif secim == "3":
            print("Program sonlandırıldı.")
            break
        else:
            print("Geçersiz seçim.")
