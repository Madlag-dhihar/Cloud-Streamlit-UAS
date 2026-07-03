import os
import json
import joblib
import pandas as pd

def model_fn(model_dir):
    print("Loading model artifacts from:", model_dir)
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    mlb = joblib.load(os.path.join(model_dir, "mlb.pkl"))
    
    return {
        "model": model, 
        "preprocessor": preprocessor, 
        "mlb": mlb
    }

def input_fn(request_body, request_content_type):
    print(f"Menerima request dengan tipe konten: {request_content_type}")
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        return pd.DataFrame(data["instances"])
    else:
        raise ValueError(f"Tipe konten '{request_content_type}' tidak didukung oleh endpoint ini.")

def predict_fn(input_data, model_artifacts):
    print("Memulai proses transformasi data dan prediksi...")
    
    
    model = model_artifacts["model"]
    preprocessor = model_artifacts["preprocessor"]
    mlb = model_artifacts["mlb"]

    loan_lists = input_data['Type_of_Loan'].tolist()
    loan_encoded = mlb.transform(loan_lists)
    
    df_loan = pd.DataFrame(loan_encoded, columns=mlb.classes_, index=input_data.index)
    
    
    input_data = pd.concat([input_data.drop('Type_of_Loan', axis=1), df_loan], axis=1)
    
    
    input_scaled = preprocessor.transform(input_data)
    
    
    predictions = model.predict(input_scaled)
    
    
    mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
    result = [mapping[pred] for pred in predictions]
    
    print(f"Prediksi berhasil dilakukan. Hasil: {result}")
    return result

def output_fn(prediction, accept):
    print(f"Mengirimkan output kembali dengan format accept: {accept}")
    
    if accept == "application/json":
        return json.dumps({"predictions": prediction}), accept
    raise ValueError(f"Tipe accept '{accept}' tidak didukung.")