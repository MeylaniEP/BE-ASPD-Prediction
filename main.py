from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # <<-- IMPORT METRIK

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="Model Prediction API",
    description="API untuk memprediksi hasil menggunakan model Random Forest dan Decision Tree.",
    version="1.0.0"
)

# --- Konfigurasi CORS ---
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model Prediksi ---
model_rf = None
model_dt = None

try:
    model_rf = joblib.load('best_rf_model_ke5.pkl')
    print("Random Forest model loaded successfully!")
except FileNotFoundError:
    print("Error: 'best_rf_model_ke5.pkl' not found. Make sure it's in the same directory as main.py.")
except Exception as e:
    print(f"Error loading Random Forest model: {e}")

try:
    model_dt = joblib.load('best_dt_model_ke5.pkl')
    print("Decision Tree model loaded successfully!")
except FileNotFoundError:
    print("Error: 'best_dt_model_ke5.pkl' not found. Make sure it's in the same directory as main.py.")
except Exception as e:
    print(f"Error loading Decision Tree model: {e}")

# --- Definisi Struktur Data Input (Pydantic Model) ---
class PredictionInput(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_by_alias=True
    )

    aspd_numerasi: float
    aspd_sains: float
    bahasa_indonesia1: float
    aspd_literasi: float
    rata_rata_literasi1: float
    bahasa_indonesia2: float
    nilai_aktual: float | None = None # <<-- FIELD BARU: Nilai Aktual (opsional)

# --- Mapping dari nama Pydantic (snake_case) ke nama Kolom Model Asli ---
PYDANTIC_TO_MODEL_COL_MAPPING = {
    "aspd_numerasi": "ASPD Numerasi",
    "aspd_sains": "ASPD Sains",
    "bahasa_indonesia1": "bahasa indonesia1",
    "aspd_literasi": "ASPD Literasi",
    "rata_rata_literasi1": "rata-rata Literasi1",
    "bahasa_indonesia2": "bahasa indonesia2",
}

# Urutan kolom yang diharapkan model (dari error message sebelumnya)
EXPECTED_MODEL_COLUMNS_ORDER = [
    'bahasa indonesia1',
    'rata-rata Literasi1',
    'ASPD Literasi',
    'bahasa indonesia2',
    'ASPD Numerasi',
    'ASPD Sains'
]

# --- Fungsi Pembantu untuk Melakukan Prediksi dan Menghitung Metrik ---
def make_prediction_and_evaluate(model, input_df, actual_value: float | None = None):
    prediction = model.predict(input_df).tolist()[0]
    metrics = None
    if actual_value is not None:
        try:
            # Karena input_df hanya berisi 1 sampel, prediksi juga 1 sampel
            # y_true dan y_pred harus berupa list/array
            y_true = [actual_value]
            y_pred = [prediction]

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            # R2 score dengan satu sampel mungkin tidak bermakna atau bahkan error jika y_true konstan
            # Periksa jika ada variasi di y_true untuk R2
            r2 = r2_score(y_true, y_pred) if np.std(y_true) > 0 else None

            metrics = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
        except Exception as e:
            print(f"Warning: Could not calculate metrics. Error: {e}")
            metrics = {"error": f"Could not calculate metrics: {e}"}
            
    return {"prediction": prediction, "metrics": metrics}


# --- Endpoint API ---

@app.get("/", summary="Root Endpoint")
async def read_root():
    return {"message": "Welcome to the Model Prediction API! Access /docs for Swagger UI."}

# <<-- ENDPOINT BARU UNTUK PREDIKSI SEMUA MODEL -->>
@app.post("/predict_all", summary="Prediksi dengan Semua Model dan Hitung Metrik")
async def predict_all(data: PredictionInput):
    rf_results = None
    dt_results = None

    try:
        input_data_dict = data.model_dump()
        actual_value = input_data_dict.pop("nilai_aktual", None) # Ambil nilai_aktual dan hapus dari dict input
        
        mapped_data = {
            PYDANTIC_TO_MODEL_COL_MAPPING[key]: value
            for key, value in input_data_dict.items()
        }
        input_df = pd.DataFrame([mapped_data])

        # --- Prediksi dengan Random Forest ---
        if model_rf is None:
            raise HTTPException(status_code=500, detail="Random Forest model not loaded.")
        
        rf_input_df = input_df.copy() # Buat salinan agar tidak mempengaruhi urutan kolom model lain
        if hasattr(model_rf, 'feature_names_in_') and model_rf.feature_names_in_ is not None:
             rf_input_df = rf_input_df[list(model_rf.feature_names_in_)]
        else:
             rf_input_df = rf_input_df[EXPECTED_MODEL_COLUMNS_ORDER]
        
        rf_results = make_prediction_and_evaluate(model_rf, rf_input_df, actual_value)


        # --- Prediksi dengan Decision Tree ---
        if model_dt is None:
            raise HTTPException(status_code=500, detail="Decision Tree model not loaded.")
        
        dt_input_df = input_df.copy() # Buat salinan
        if hasattr(model_dt, 'feature_names_in_') and model_dt.feature_names_in_ is not None:
            dt_input_df = dt_input_df[list(model_dt.feature_names_in_)]
        else:
            dt_input_df = dt_input_df[EXPECTED_MODEL_COLUMNS_ORDER]
        
        dt_results = make_prediction_and_evaluate(model_dt, dt_input_df, actual_value)

        return {
            "rf_results": rf_results,
            "dt_results": dt_results
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {ve}. Please check data types and values.")
    except KeyError as ke:
        raise HTTPException(status_code=422, detail=f"Error mapping input features: {ke}. Check PYDANTIC_TO_MODEL_COL_MAPPING and input data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# Opsional: Anda bisa menghapus endpoint /predict_rf dan /predict_dt yang lama
# jika Anda hanya ingin menggunakan endpoint /predict_all

# @app.post("/predict_rf", summary="Prediksi dengan Random Forest Model (Lama)")
# async def predict_rf_old(data: PredictionInput):
#     # ... kode lama ...
#     pass

# @app.post("/predict_dt", summary="Prediksi dengan Decision Tree Model (Lama)")
# async def predict_dt_old(data: PredictionInput):
#     # ... kode lama ...
#     pass