Akademik kadın 🌸 — aşağıda hem senin **`aciklama (1).md`** dosyandaki açıklamaları hem de **`train.py`** dosyandaki MLflow işlemlerini birleştirerek profesyonel, teslim edilebilir bir **`README.md`** dosyası oluşturdum.

---

# 📘 MLflow + Scikit-learn Regresyon Projesi

## 🎯 Proje Amacı

Bu proje, makine öğrenmesi regresyon sürecinin uçtan uca yönetimini **MLflow** platformu ile entegre ederek gerçekleştirmeyi amaçlar.
Adımlar sırasıyla:

1. Veri kümesinin yüklenmesi ve ön işlenmesi
2. scikit-learn ile regresyon modelinin eğitilmesi
3. Parametrelerin, metriklerin ve grafiklerin **MLflow’a kaydedilmesi**
4. Eğitilen modelin MLflow’a **artifact ve registered model** olarak yüklenmesi
5. Performans değerlendirmesi

---

## ⚙️ Kullanılan Kütüphaneler

| Kütüphane                 | Görevi                                                   |
| ------------------------- | -------------------------------------------------------- |
| **mlflow**                | Deney yönetimi, metrik, parametre ve model takibi        |
| **scikit-learn**          | Modelleme, metrik hesaplama, veri bölme ve ölçeklendirme |
| **numpy, pandas**         | Sayısal işlemler ve veri analizi                         |
| **matplotlib**            | Tahmin–gerçek grafiklerinin çizimi                       |
| **argparse**              | Terminal üzerinden parametre alımı                       |
| **logging, warnings, os** | Loglama, hata yönetimi, dosya işlemleri                  |

---

## 📦 Veri Kümesi

Kod şu anda **California Housing** veri setini kullanır:

* Özellikler: Oda sayısı, gelir, konum bilgisi vb.
* Hedef: `MedHouseVal` (ev fiyatı, 100.000$ cinsinden)
  Veri otomatik olarak `fetch_california_housing()` fonksiyonu ile indirilir.


---

## 🧠 Model Türleri

İki model seçeneği bulunmaktadır:

1. **Linear Regression (`--model linreg`)** – Basit doğrusal model
2. **Random Forest Regressor (`--model rf`)** – Ağaç tabanlı topluluk modeli

Random Forest için ayarlanabilir hiperparametreler:

* `--n_estimators`: Ağaç sayısı
* `--max_depth`: Maksimum derinlik
* `--min_samples_split`: Dallanma için minimum örnek sayısı

---

## 📊 MLflow Süreci ve Loglama Adımları

### 1️⃣ Deney (Experiment) ve Run Başlatma

```python
mlflow.set_experiment(args.experiment)
with mlflow.start_run():
```

Yeni bir “deney” (experiment) başlatılır ve her eğitim oturumu “run” olarak kaydedilir.

### 2️⃣ Parametrelerin Loglanması

```python
mlflow.log_params({...})
```

Model tipi, test oranı, ölçeklendirme durumu, hiperparametreler (`n_estimators`, `max_depth`, vb.) MLflow’a kaydedilir.

### 3️⃣ Metriklerin Loglanması

```python
mlflow.log_metrics({
    "test_rmse": rmse_test,
    "test_mae": mae_test,
    "test_r2": r2_test,
    "train_rmse": rmse_train,
    "train_mae": mae_train,
    "train_r2": r2_train
})
```

Eğitim ve test sonuçları (RMSE, MAE, R²) MLflow UI üzerinden izlenebilir.

### 4️⃣ Grafiklerin Artifact Olarak Kaydedilmesi

```python
plot_predictions(y_test, y_pred, "artifacts/pred_vs_true.png")
mlflow.log_artifact("artifacts/pred_vs_true.png")
```

Gerçek vs Tahmin grafiği `artifacts/` klasörüne kaydedilip MLflow’a yüklenir.
Kırmızı kesikli çizgi ideal tahmini, mavi noktalar model tahminlerini gösterir.

### 5️⃣ Modelin MLflow’a Kaydedilmesi

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name=args.register_name,
    input_example=X_train.iloc[:5].to_dict('records')
)
```

Model hem **artifact** olarak hem de **Registered Model** olarak MLflow’da saklanır.
Arayüzde “Models” sekmesinden versiyonlama yapılabilir.

---

## 📈 Konsol Çıktısı ve Performans Analizi

Model eğitimi tamamlandığında terminalde şu tablo görüntülenir:

```
METRIK               TEST            TRAIN
RMSE                 0.7345          0.6901
MAE                  0.5253          0.4900
R² Score             0.8241          0.8312
```

Ek olarak:

* R² ≥ 0.85 → ✅ Mükemmel performans
* 0.7 ≤ R² < 0.85 → ✓ İyi performans
* R² < 0.7 → ⚠ Geliştirilebilir model

Model aşırı öğrenme (overfitting) gösteriyorsa öneriler terminalde görüntülenir:

* `--min_samples_split` artır
* `--max_depth` azalt
* `--scale` parametresini ekle

---

## Çalıştırma ve Değerlendirme Süreci

1. **Kurulum**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # (Windows)
   pip install -r requirements.txt
   ```
2. **Model Eğitimi**

   ```bash
   python src/train.py --model rf --n_estimators 100 --max_depth 12 --scale
   ```
3. **MLflow Arayüzünü Açmak**

   ```bash
   mlflow ui --backend-store-uri ./mlruns --port 5000
   ```

   Tarayıcıdan `http://127.0.0.1:5000` adresine gidin.

   * **Experiments** sekmesinde koşular (runs)
   * **Metrics**, **Params**, **Artifacts** sekmelerinde detaylı sonuçlar
   * **Models** sekmesinde kayıtlı modelleri görebilirsiniz.
4. **Kaydedilmiş Modeli Yüklemek**

   ```python
   import mlflow.sklearn
   model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
   pred = model.predict([...])
   ```
5. **Hata Durumunda**

   * “MemoryError”: Büyük veri setlerinde `test_size` oranını artırın (örn. `--test_size 0.3`)
   * “Port already in use”: `mlflow ui --port 5001` komutuyla başka port seçin

---

## ✅ Sonuç

| Adım | İşlem            | Açıklama                                     |
| ---- | ---------------- | -------------------------------------------- |
| 1    | MLflow kurulumu  | Deney oluşturma, parametre ve metrik kaydı   |
| 2    | Veri yükleme     | California Housing (veya NYC Taxi) veri seti |
| 3    | Model eğitimi    | Linear Regression veya Random Forest         |
| 4    | Metrik hesaplama | RMSE, MAE, R²                                |
| 5    | MLflow’a kayıt   | Model, metrik, parametre ve grafik loglama   |
| 6    | Raporlama        | Performans analizi      |

---

Bu proje, MLflow’un deney izleme, model versiyonlama ve performans analizi kabiliyetlerini **scikit-learn tabanlı regresyon** üzerinde gösteren eksiksiz bir pipeline’dır.

---
