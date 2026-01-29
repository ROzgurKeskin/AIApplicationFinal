
import json
import pickle
import os
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_scheduler
)
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset

# === Eğitimden mi başlasın? ===
TRAIN_FROM_SCRATCH = False  # True: Baştan eğit, False: Mevcut modelleri yükle
# === Ayarlar ===
basePath = "C:\\PythonProject\\2.donem projeleri\\YapayZekaninPrensipleri\\FinalProject"
modelPath = "C:\\PythonProject\\2.donem projeleri\\YapayZekaninPrensipleri\\FinalProject\\model"
resultSavingPath = "C:\\PythonProject\\2.donem projeleri\\YapayZekaninPrensipleri\\FinalProject\\Results\\images"
modelFolder = "cross"
intentPath = "C:\\PythonProject\\2.donem projeleri\YapayZekaninPrensipleri\\FinalProject\\intents.json"
turkishModelPath = "dbmdz/bert-base-turkish-cased"
startTime = datetime.now()

sentences = []

# === Veriyi Yükle ve Dönüştür ===
with open(intentPath, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
main_labels = []
sub_labels = []

for intent in data["intents"]:
    main_cat = intent["MainCategory"]
    for item in intent["Items"]:
        sub_cat = item["SubCategory"]
        for pattern in item["Patterns"]:
            if pattern.strip():
                sentences.append(pattern)
                main_labels.append(main_cat)
                sub_labels.append(f"{main_cat}__{sub_cat}")

# Subcategory modelini her ana kategori için ayrı eğitmek için mapping hazırla
subcat_sentences = defaultdict(list)
subcat_labels = defaultdict(list)
for s, m, sub in zip(sentences, main_labels, sub_labels):
    subcat_sentences[m].append(s)
    subcat_labels[m].append(sub)



# === ANA KATEGORİ MODELİ ===
main_le = LabelEncoder()
main_labels_enc = main_le.fit_transform(main_labels)
print("MainCategory Sınıf Dağılımı:", Counter(main_labels))

# Tokenizer
tokenizer_path = f"{modelPath}/{modelFolder}"
if os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
else:
    tokenizer = BertTokenizer.from_pretrained(turkishModelPath)

main_encodings = tokenizer(list(sentences), truncation=True, padding='max_length', max_length=128)
main_dataset = Dataset.from_dict({
    "input_ids": main_encodings["input_ids"],
    "attention_mask": main_encodings["attention_mask"],
    "labels": main_labels_enc
})

main_model = BertForSequenceClassification.from_pretrained(turkishModelPath, num_labels=len(main_le.classes_))

# Sınıf ağırlıkları
main_class_counts = Counter(main_labels)
total_samples = sum(main_class_counts.values())
main_class_weights = {label: total_samples / count for label, count in main_class_counts.items()}
main_class_weights_list = [main_class_weights[main_le.inverse_transform([i])[0]] for i in range(len(main_le.classes_))]
main_class_weights_tensor = torch.tensor(main_class_weights_list, dtype=torch.float32)
main_loss_fn = CrossEntropyLoss(weight=main_class_weights_tensor)

def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


main_training_args = TrainingArguments(
    output_dir=f"{modelPath}/{modelFolder}/main_category",
    num_train_epochs=3,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=1,
    logging_dir=f"{basePath}\\logs",
    logging_steps=10,
    warmup_steps=500
)

main_trainer = CustomTrainer(
    model=main_model,
    args=main_training_args,
    train_dataset=main_dataset,
    compute_metrics=compute_metrics,
    loss_fn=main_loss_fn
)

main_trainer.train()

with open(f"{modelPath}/{modelFolder}/main_label_encoder.pkl", "wb") as f:
    pickle.dump(main_le, f)
main_eval_results = main_trainer.evaluate(eval_dataset=main_dataset)
print(f"Ana kategori başarı oranı: {main_eval_results.get('eval_accuracy', 'N/A')}")

# === ALT KATEGORİ MODELLERİ ===
subcat_models = {}
subcat_label_encoders = {}
for main_cat in subcat_sentences:
    sub_le = LabelEncoder()
    sub_labels_enc = sub_le.fit_transform(subcat_labels[main_cat])
    subcat_label_encoders[main_cat] = sub_le
    encodings = tokenizer(subcat_sentences[main_cat], truncation=True, padding='max_length', max_length=128)
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": sub_labels_enc
    })
    class_counts = Counter(subcat_labels[main_cat])
    total_samples = sum(class_counts.values())
    class_weights = {label: total_samples / count for label, count in class_counts.items()}
    class_weights_list = [class_weights[sub_le.inverse_transform([i])[0]] for i in range(len(sub_le.classes_))]
    class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32)
    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)
    model = BertForSequenceClassification.from_pretrained(turkishModelPath, num_labels=len(sub_le.classes_))
    training_args = TrainingArguments(
        output_dir=f"{modelPath}/{modelFolder}/subcat_{main_cat}",
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_total_limit=1,
        logging_dir=f"{basePath}\\logs",
        logging_steps=10,
        warmup_steps=500
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics,
        loss_fn=loss_fn
    )
    trainer.train()
    subcat_models[main_cat] = model
    with open(f"{modelPath}/{modelFolder}/subcat_label_encoder_{main_cat}.pkl", "wb") as f:
        pickle.dump(sub_le, f)
    subcat_eval_results = trainer.evaluate(eval_dataset=dataset)
    print(f"Alt kategori ({main_cat}) başarı oranı: {subcat_eval_results.get('eval_accuracy', 'N/A')}")

print("Tüm ana ve alt kategori modelleri eğitildi.")

# === Eğitimden mi başlasın kontrolü ===
if not TRAIN_FROM_SCRATCH:
    # Ana kategori modeli yükle
    from transformers import BertForSequenceClassification
    import pickle
    main_model = BertForSequenceClassification.from_pretrained(f"{modelPath}/{modelFolder}/main_category")
    with open(f"{modelPath}/{modelFolder}/main_label_encoder.pkl", "rb") as f:
        main_le = pickle.load(f)
    # Ana kategori veri setini oluştur
    encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=128)
    main_labels_enc = main_le.transform(main_labels)
    main_dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": main_labels_enc
    })
    main_trainer = CustomTrainer(
        model=main_model,
        args=main_training_args,
        train_dataset=main_dataset,
        compute_metrics=compute_metrics,
        loss_fn=main_loss_fn
    )
    main_eval_results = main_trainer.evaluate(eval_dataset=main_dataset)
    print(f"Ana kategori başarı oranı (YÜKLENEN): {main_eval_results.get('eval_accuracy', 'N/A')}")

    # Alt kategori modellerini yükle ve başarı oranlarını yaz
    for main_cat in subcat_sentences:
        model = BertForSequenceClassification.from_pretrained(f"{modelPath}/{modelFolder}/subcat_{main_cat}")
        with open(f"{modelPath}/{modelFolder}/subcat_label_encoder_{main_cat}.pkl", "rb") as f:
            sub_le = pickle.load(f)
        encodings = tokenizer(subcat_sentences[main_cat], truncation=True, padding='max_length', max_length=128)
        sub_labels_enc = sub_le.transform(subcat_labels[main_cat])
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": sub_labels_enc
        })
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=compute_metrics,
            loss_fn=loss_fn
        )
        subcat_eval_results = trainer.evaluate(eval_dataset=dataset)
        print(f"Alt kategori ({main_cat}) başarı oranı (YÜKLENEN): {subcat_eval_results.get('eval_accuracy', 'N/A')}")

# Inference örneği:
# 1. main_pred = main_model.predict(sentence)
# 2. subcat_pred = subcat_models[main_pred].predict(sentence)



