from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
model_rf_ke5 = None
model_dt_ke5 = None
model_rf_baseline = None # Changed variable name
model_dt_baseline = None # Changed variable name

try:
    model_rf_ke5 = joblib.load('best_rf_model_ke5.pkl')
    print("Random Forest model (ke5) loaded successfully!")
except FileNotFoundError:
    print("Error: 'best_rf_model_ke5.pkl' not found. Make sure it's in the same directory as main.py.")
except Exception as e:
    print(f"Error loading Random Forest model (ke5): {e}")

try:
    model_dt_ke5 = joblib.load('best_dt_model_ke5.pkl')
    print("Decision Tree model (ke5) loaded successfully!")
except FileNotFoundError:
    print("Error: 'best_dt_model_ke5.pkl' not found. Make sure it's in the same directory as main.py.")
except Exception as e:
    print(f"Error loading Decision Tree model (ke5): {e}")

# Load the new models (now baseline models)
try:
    model_rf_baseline = joblib.load('baseline_rf_model.pkl') # Changed file name
    print("Random Forest model (baseline) loaded successfully!")
except FileNotFoundError:
    print("Error: 'baseline_rf_model.pkl' not found. Make sure it's in the same directory as main.py.")
except Exception as e:
    print(f"Error loading Random Forest model (baseline): {e}")

try:
    model_dt_baseline = joblib.load('baseline_dt_model.pkl') # Changed file name
    print("Decision Tree model (baseline) loaded successfully!")
except FileNotFoundError:
    print("Error: 'baseline_dt_model.pkl' not found. Make sure it's in the same directory as main.py.")
except Exception as e:
    print(f"Error loading Decision Tree model (baseline): {e}")

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
    nilai_aktual: float | None = None

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
    rf_ke5_results = None
    dt_ke5_results = None
    rf_baseline_results = None # Changed variable name
    dt_baseline_results = None # Changed variable name

    try:
        input_data_dict = data.model_dump()
        actual_value = input_data_dict.pop("nilai_aktual", None) # Ambil nilai_aktual dan hapus dari dict input
        
        mapped_data = {
            PYDANTIC_TO_MODEL_COL_MAPPING[key]: value
            for key, value in input_data_dict.items()
        }
        input_df = pd.DataFrame([mapped_data])

        # --- Prediksi dengan Random Forest ke5 ---
        if model_rf_ke5 is None:
            raise HTTPException(status_code=500, detail="Random Forest model (ke5) not loaded.")
        
        rf_ke5_input_df = input_df.copy() # Buat salinan agar tidak mempengaruhi urutan kolom model lain
        if hasattr(model_rf_ke5, 'feature_names_in_') and model_rf_ke5.feature_names_in_ is not None:
              rf_ke5_input_df = rf_ke5_input_df[list(model_rf_ke5.feature_names_in_)]
        else:
              rf_ke5_input_df = rf_ke5_input_df[EXPECTED_MODEL_COLUMNS_ORDER]
        
        rf_ke5_results = make_prediction_and_evaluate(model_rf_ke5, rf_ke5_input_df, actual_value)

        # --- Prediksi dengan Decision Tree ke5 ---
        if model_dt_ke5 is None:
            raise HTTPException(status_code=500, detail="Decision Tree model (ke5) not loaded.")
        
        dt_ke5_input_df = input_df.copy() # Buat salinan
        if hasattr(model_dt_ke5, 'feature_names_in_') and model_dt_ke5.feature_names_in_ is not None:
            dt_ke5_input_df = dt_ke5_input_df[list(model_dt_ke5.feature_names_in_)]
        else:
            dt_ke5_input_df = dt_ke5_input_df[EXPECTED_MODEL_COLUMNS_ORDER]
        
        dt_ke5_results = make_prediction_and_evaluate(model_dt_ke5, dt_ke5_input_df, actual_value)

        # --- Prediksi dengan Random Forest Baseline ---
        if model_rf_baseline is None: # Changed variable name
            raise HTTPException(status_code=500, detail="Random Forest model (baseline) not loaded.")
        
        rf_baseline_input_df = input_df.copy() # Changed variable name
        if hasattr(model_rf_baseline, 'feature_names_in_') and model_rf_baseline.feature_names_in_ is not None: # Changed variable name
              rf_baseline_input_df = rf_baseline_input_df[list(model_rf_baseline.feature_names_in_)] # Changed variable name
        else:
              rf_baseline_input_df = rf_baseline_input_df[EXPECTED_MODEL_COLUMNS_ORDER]
        
        rf_baseline_results = make_prediction_and_evaluate(model_rf_baseline, rf_baseline_input_df, actual_value) # Changed variable name

        # --- Prediksi dengan Decision Tree Baseline ---
        if model_dt_baseline is None: # Changed variable name
            raise HTTPException(status_code=500, detail="Decision Tree model (baseline) not loaded.")
        
        dt_baseline_input_df = input_df.copy() # Changed variable name
        if hasattr(model_dt_baseline, 'feature_names_in_') and model_dt_baseline.feature_names_in_ is not None: # Changed variable name
            dt_baseline_input_df = dt_baseline_input_df[list(model_dt_baseline.feature_names_in_)] # Changed variable name
        else:
            dt_baseline_input_df = dt_baseline_input_df[EXPECTED_MODEL_COLUMNS_ORDER]
        
        dt_baseline_results = make_prediction_and_evaluate(model_dt_baseline, dt_baseline_input_df, actual_value) # Changed variable name

        return {
            "rf_ke5_results": rf_ke5_results,
            "dt_ke5_results": dt_ke5_results,
            "rf_baseline_results": rf_baseline_results, # Changed key in response
            "dt_baseline_results": dt_baseline_results  # Changed key in response
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {ve}. Please check data types and values.")
    except KeyError as ke:
        raise HTTPException(status_code=422, detail=f"Error mapping input features: {ke}. Check PYDANTIC_TO_MODEL_COL_MAPPING and input data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")