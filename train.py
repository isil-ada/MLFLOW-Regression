import argparse
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışır
import matplotlib.pyplot as plt 
import warnings
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Tüm uyarıları kapat
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(test_size=0.2, random_state=42, scale=False):
    """
    California Housing veri setini yükler ve ön işleme yapar.
    
    Args:
        test_size: Test seti oranı
        random_state: Rastgelelik için seed değeri
        scale: Standart ölçeklendirme yapılıp yapılmayacağı
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        print("Veri seti yükleniyor...")
        
        # California Housing veri seti
        data = fetch_california_housing(as_frame=True)
        X = data.frame.copy()
        y = data.target.copy()
        
        print(f"Veri seti boyutu: {X.shape}")
        print(f"Özellikler: {X.columns.tolist()}")
        print(f"Hedef değişken: MedHouseVal (Ev fiyatı - 100k$ cinsinden)")
        
        # Eksik değerleri kontrol et
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        # Sonsuz değerleri temizle
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        if scale:
            print("Özellikler ölçeklendiriliyor...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Eğitim seti: {X_train.shape[0]} örnek")
        print(f"Test seti: {X_test.shape[0]} örnek")
        print(f"Hedef değişken aralığı: {y.min():.2f} - {y.max():.2f}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        raise

def get_model(name, args):
    """
    Belirtilen modeli ve parametrelerini döndürür.
    
    Args:
        name: Model adı ('linreg' veya 'rf')
        args: Argparse argümanları
    
    Returns:
        model, params (tuple)
    """
    if name == "linreg":
        model = LinearRegression(n_jobs=-1)
        params = {"model": name}
        print("Model: Linear Regression")
    elif name == "rf":
        max_depth_val = args.max_depth if args.max_depth > 0 else None
        min_samples_val = max(2, args.min_samples_split)
        
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=max_depth_val,
            min_samples_split=min_samples_val,
            random_state=args.random_state,
            n_jobs=-1,
            verbose=0,
            max_features='sqrt',  # Overfitting'i azaltır
            warm_start=False
        )
        params = {
            "model": name,
            "n_estimators": args.n_estimators,
            "max_depth": str(max_depth_val),
            "min_samples_split": min_samples_val,
            "random_state": args.random_state
        }
        print(f"Model: Random Forest (n_estimators={args.n_estimators}, max_depth={max_depth_val})")
    else:
        raise ValueError("Model 'linreg' veya 'rf' olmalıdır!")
    
    return model, params

def plot_predictions(y_true, y_pred, out_path):
    """
    Gerçek ve tahmin edilen değerleri görselleştirir.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        out_path: Çıktı dosya yolu
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.3, s=10, c='blue', edgecolors='none')
        plt.xlabel("Gerçek Değer", fontsize=12)
        plt.ylabel("Tahmin", fontsize=12)
        plt.title("Gerçek vs Tahmin (Test Seti)", fontsize=14)
        
        min_v = float(min(y_true.min(), y_pred.min()))
        max_v = float(max(y_true.max(), y_pred.max()))
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='İdeal Tahmin')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close('all')
        print(f"✓ Grafik kaydedildi: {out_path}")
    except Exception as e:
        print(f"⚠ Grafik oluşturma hatası (görmezden gelindi): {e}")

def main():
    parser = argparse.ArgumentParser(description="MLflow ile Regresyon Modeli Eğitimi")
    parser.add_argument("--experiment", type=str, default="regression-demo", 
                        help="MLflow deney adı")
    parser.add_argument("--model", type=str, default="rf", choices=["linreg", "rf"],
                        help="Model tipi: 'linreg' veya 'rf'")
    parser.add_argument("--scale", action="store_true",
                        help="Özellikleri standart ölçeklendirme uygula")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test seti oranı")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Rastgelelik seed değeri")
    parser.add_argument("--n_estimators", type=int, default=50,
                        help="Random Forest için ağaç sayısı")
    parser.add_argument("--max_depth", type=int, default=8,
                        help="Random Forest için maksimum derinlik (0=sınırsız)")
    parser.add_argument("--min_samples_split", type=int, default=10,
                        help="Bölünme için minimum örnek sayısı")
    parser.add_argument("--register_name", type=str, default="california-housing-regressor",
                        help="Model kayıt adı")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "MLflow Regresyon Pipeline")
    print("=" * 70)
    print(f"Model: {args.model.upper()}")
    print(f"Deney: {args.experiment}")
    print(f"Ölçeklendirme: {'Evet' if args.scale else 'Hayır'}")
    print(f"Test Oranı: {args.test_size}")
    print("=" * 70 + "\n")
    
    try:
        # MLflow deneyini ayarla
        mlflow.set_experiment(args.experiment)
        
        with mlflow.start_run():
            # Veriyi yükle
            X_train, X_test, y_train, y_test = load_data(
                test_size=args.test_size,
                random_state=args.random_state,
                scale=args.scale
            )
            
            # Modeli al
            model, params = get_model(args.model, args)
            
            # Parametreleri logla
            mlflow.log_params({
                "scale": args.scale,
                "test_size": args.test_size,
                **params
            })
            
            # Modeli eğit
            print("\n" + "-" * 70)
            print("Model eğitiliyor...")
            model.fit(X_train, y_train)
            print("✓ Model eğitimi tamamlandı!")
            print("-" * 70)
            
            # Tahmin yap ve metrikları hesapla
            print("\nTahminler yapılıyor...")
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Test metrikleri
            rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae_test = float(mean_absolute_error(y_test, y_pred_test))
            r2_test = float(r2_score(y_test, y_pred_test))
            
            # Eğitim metrikleri
            rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
            mae_train = float(mean_absolute_error(y_train, y_pred_train))
            r2_train = float(r2_score(y_train, y_pred_train))
            
            # Metrikleri logla
            mlflow.log_metrics({
                "test_rmse": rmse_test,
                "test_mae": mae_test,
                "test_r2": r2_test,
                "train_rmse": rmse_train,
                "train_mae": mae_train,
                "train_r2": r2_train
            })
            
            # Görselleştirme
            os.makedirs("artifacts", exist_ok=True)
            plot_path = "artifacts/pred_vs_true.png"
            plot_predictions(y_test.reset_index(drop=True), pd.Series(y_pred_test), plot_path)
            
            try:
                mlflow.log_artifact(plot_path)
            except:
                pass  # Artifact yükleme hatası görmezden geliniyor
            
            # Modeli kaydet
            print("\nModel MLflow'a kaydediliyor...")
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=args.register_name,
                    input_example=X_train.iloc[:5].to_dict('records')
                )
                print("✓ Model kaydedildi!")
            except Exception as e:
                print(f"⚠ Model kayıt uyarısı (görmezden gelinir): {e}")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )
                print("✓ Model artifact olarak kaydedildi!")
            
            # Sonuçları yazdır
            print("\n" + "=" * 70)
            print(" " * 25 + "EĞİTİM SONUÇLARI")
            print("=" * 70)
            print(f"\n{'METRIK':<20} {'TEST':<15} {'TRAIN':<15}")
            print("-" * 70)
            print(f"{'RMSE':<20} {rmse_test:<15.4f} {rmse_train:<15.4f}")
            print(f"{'MAE':<20} {mae_test:<15.4f} {mae_train:<15.4f}")
            print(f"{'R² Score':<20} {r2_test:<15.4f} {r2_train:<15.4f}")
            print("=" * 70)
            
            # Performans değerlendirmesi
            print("\n💡 MODEL PERFORMANSI:")
            if r2_test > 0.85:
                print("   ✅ Mükemmel performans!")
            elif r2_test > 0.7:
                print("   ✓ İyi performans")
            else:
                print("   ⚠ Performans geliştirilebilir:")
                print("      - Daha fazla ağaç deneyin: --n_estimators 100")
                print("      - Daha derin ağaçlar: --max_depth 15")
                print("      - Ölçeklendirme ekleyin: --scale")
            
            # Overfitting kontrolü
            overfitting_ratio = rmse_train / rmse_test if rmse_test > 0 else 1
            if overfitting_ratio < 0.8:
                print("\n💡 NOT: Hafif overfitting görülüyor (genellikle normaldir)")
                print("   İyileştirme için:")
                print("      - min_samples_split artırın: --min_samples_split 20")
                print("      - max_depth azaltın: --max_depth 6")
            
            print("\n✅ Pipeline başarıyla tamamlandı!\n")
            
    except Exception as e:
        print(f"\n❌ HATA OLUŞTU: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()