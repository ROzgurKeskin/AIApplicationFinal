import pandas as pd
import time
import os
import requests
import json

# 1. API Yapılandırması (Kendi anahtarını buraya yapıştır)
GOOGLE_API_KEY = "AIzaSyCF60H1StSsV8rPAYqoz31hhNKTSRkx7_w"

def get_llm_category(body_text):
    categories = [
        "Bağlantı sorunu", "Donanım talebi", "Erişim hatası", "Erişim talebi",
        "Hata mesajı alıyorum", "Mail grubu ekleme", "Rapor talebi", "Sistem hatası",
        "Sunucuya ulaşılamıyor", "Uygulama çalışmıyor", "VPN bağlantı problemi",
        "VPN erişim isteği", "Yazılım kurulumu", "Yeni kullanıcı isteği", "Yetki talebi"
    ]
    
    # Senin sisteminde AKTİF görünen tam isim: gemini-2.0-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GOOGLE_API_KEY}"
    
    headers = {'Content-Type': 'application/json'}
    prompt = f"Sen bir IT Destek Uzmanısın. Sadece listeden EN UYGUN kategoriyi seç, açıklama yapma: {', '.join(categories)}\n\nMesaj: {body_text}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        res_json = response.json()
        
        if response.status_code == 200:
            result = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Model cevabını listedeki kategorilerle eşle
            for cat in categories:
                if cat.lower() in result.lower():
                    return cat
            return result
        elif response.status_code == 429:
            return "KOTA_LIMITI"
        else:
            # Hatanın gerçek nedenini döndür
            return f"Hata: {res_json.get('error', {}).get('message', 'Bilinmeyen')[:15]}"
    except Exception as e:
        return f"Bağlantı: {str(e)[:15]}"

def run_llm_phase():
    input_path = "data/processed/regex_results.xlsx" 
    output_path = "data/processed/final_submission_result.xlsx"
    
    df = pd.read_excel(input_path)
    mask = df['label'] == 'BELİRSİZ'
    undetermined_rows = df[mask]
    
    print(f"\n🚀 ANALİZ BAŞLADI... (Toplam: {len(undetermined_rows)} kayıt)")

    for index, row in undetermined_rows.iterrows():
        label = get_llm_category(row['body'])
        
        # Kota hatası durumunda 60 saniye bekle ve aynı satırı tekrar dene
        if "KOTA_LIMITI" in label:
            print(f"\n[!] Kota doldu, 5 sn bekleniyor...", end="\r")
            time.sleep(5)
            label = get_llm_category(row['body'])

        print(f"[*] ID: {row['id']} -> {label}")
        df.at[index, 'label'] = label
        
        # Dakikada 10-12 istek için 6 saniye bekleme (En güvenli süre)
        time.sleep(6)
        
        # Her 10 satırda bir Excel'i kaydet (Elektrik gitse bile veriler kalsın)
        if index % 10 == 0:
            df.to_excel(output_path, index=False)
    
    df.to_excel(output_path, index=False)
    print("\n🎯 İŞLEM TAMAMLANDI! Sonuçlar: " + output_path)

if __name__ == "__main__":
    run_llm_phase()