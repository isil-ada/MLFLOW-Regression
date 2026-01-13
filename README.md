---

# MLflow ile Regresyon Projesi

Bu proje, makine öğrenimi regresyon modelleme sürecinin **uçtan uca yönetimini MLflow ile entegre ederek** göstermeyi amaçlayan bir çalışmadır. Çalışma, **ders kapsamında yapılmıştır** ve MLflow’un deney izleme, model kayıt ve değerlendirme özelliklerini kullanmaktadır.

---

## Proje Amacı

* MLflow ile regresyon modellerinin eğitimi, takibi ve değerlendirmesini yapmak.
* Model parametrelerini, performans metriklerini, artefaktları ve modelleri MLflow UI üzerinden izlemek.
* Farklı model türlerini deneyerek regresyon performansını karşılaştırmak.

---

## Kullanılan Teknolojiler

| Teknoloji         | Kullanım Amacı                            |
| ----------------- | ----------------------------------------- |
| **MLflow**        | Deney takibi, metrik loglama, model kaydı |
| **scikit-learn**  | Regresyon modelleri ve veri işlemleri     |
| **numpy, pandas** | Veri analizi ve sayısal işlemler          |
| **matplotlib**    | Grafik çizimi                             |
| **argparse**      | Komut satırı parametreleri                |
| **logging**       | Çıktı ve hata loglaması                   |

---

## Proje İçeriği

```
MLFLOW-Regression
 ┣ README.md
 ┣ train.py
 ┗ requirements.txt 
```

* **train.py** — Model eğitimi ve MLflow loglama işlemleri burada yapılır.
* **README.md** — Proje açıklaması ve kullanım bilgileri.

---

## Veri Seti

Bu projede kullanılan veri seti **California Housing** veri setidir:

* Özellikler: Ev özellikleri (oda sayısı, konum, gelir vb.)
* Hedef Değişken: **MedHouseVal** (Ev fiyatı)
  Veri scikit-learn içinden otomatik indirilir ve işlenir.

---

## Model Seçenekleri

Proje iki farklı regresyon modelini destekler:

| Model    | Açıklama                 |
| -------- | ------------------------ |
| `linreg` | Basit Doğrusal Regresyon |
| `rf`     | Random Forest Regressor  |

Random Forest için ayarlanabilir parametreler:

* `--n_estimators`
* `--max_depth`
* `--min_samples_split`

---

## MLflow Loglama Süreci

Proje içerisinde aşağıdaki adımlar MLflow’a loglanır:

1. **Experiment / Run oluşturma**
2. **Parametrelerin kaydedilmesi**
3. **Metriklerin kaydedilmesi**
4. **Grafiklerin artefakt olarak kaydedilmesi**
5. **Modelin MLflow’a artifact ve Registered Model olarak kaydedilmesi**

Model ve metrik sonuçları MLflow UI aracılığıyla görselleştirilebilir.

---

## Kurulum & Çalıştırma

Projenin çalışması için:

```bash
python -m venv .venv
source .venv/bin/activate      # (Linux/Mac)
.venv\Scripts\activate         # (Windows)

pip install -r requirements.txt
```

### Model Eğitimi

```bash
python train.py --model rf --n_estimators 100 --max_depth 12 --scale
```

veya

```bash
python train.py --model linreg --scale
```

### MLflow UI

Eğitimden sonra MLflow arayüzünü açmak için:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Tarayıcıdan: `http://127.0.0.1:5000` adresine giderek:

* **Experiments**
* **Params, Metrics, Artifacts**
* **Models**

sekmelerini inceleyebilirsin.

---

## Sonuçlar & Değerlendirme

MLflow üzerinden takip edilen deneylerde:

* RMSE, MAE ve R² gibi performans metrikleri izlenebilir.
* Model öğrenimi ve tahmin performansı grafiklerle gösterilir.
* Kaydedilen modeller versiyonlanabilir şekilde saklanır.

---

## Notlar

* Bu proje MLflow’un regresyon modelleme iş akışını göstermek amacıyla **ders kapsamında geliştirilmiştir**.
* Model performansını artırmak için veri ön işleme ve hiperparametre optimizasyonu eklenebilir.

---
