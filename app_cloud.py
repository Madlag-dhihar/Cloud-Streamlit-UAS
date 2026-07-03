import streamlit as st
import pandas as pd
import boto3
import json
import os

# Konfigurasi endpoint
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint-5")
REGION = os.environ.get("AWS_REGION", "us-east-1")

st.set_page_config(page_title="Credit Score Prediction by Shafi", layout="wide")

# Fungsi untuk memanggil SageMaker Endpoint
def query_endpoint(data):
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"instances": [data]})
    )
    result = json.loads(response["Body"].read().decode())
    return result[0] # Mengembalikan hasil prediksi

# Tampilan Streamlit tetap sama sampai bagian 'if submit_button:'
st.title("Credit Score Prediction System")
# ... (kode form kamu tetap sama) ...

if submit_button:
    # 1. Siapkan data
    input_data = {
        'Age': age, 'Annual_Income': annual_income, 'Monthly_Inhand_Salary': monthly_inhand_salary,
        'Num_Bank_Accounts': num_bank_accounts, 'Num_Credit_Card': num_credit_card,
        'Interest_Rate': interest_rate, 'Num_of_Loan': num_of_loan,
        'Delay_from_due_date': delay_from_due_date, 'Num_of_Delayed_Payment': num_delayed_payment,
        'Changed_Credit_Limit': changed_credit_limit, 'Num_Credit_Inquiries': num_credit_inquiries,
        'Outstanding_Debt': outstanding_debt, 'Credit_Utilization_Ratio': credit_util_ratio,
        'Credit_History_Age': credit_history_age, 'Total_EMI_per_month': total_emi_per_month,
        'Amount_invested_monthly': amount_invested_monthly, 'Monthly_Balance': monthly_balance,
        'Occupation': occupation, 'Credit_Mix': credit_mix,
        'Payment_Behaviour': payment_behaviour, 'Payment_of_Min_Amount': payment_of_min_amount,
        'Type_of_Loan': type_of_loan
    }
    
    # 2. Panggil SageMaker
    try:
        result_text = query_endpoint(input_data)
        
        # 3. Tampilkan hasil
        if result_text == 'Good':
            st.success(f"### Hasil Prediksi: {result_text} Credit Score")
        elif result_text == 'Standard':
            st.info(f"### Hasil Prediksi: {result_text} Credit Score")
        else:
            st.error(f"### Hasil Prediksi: {result_text} Credit Score")
            
    except Exception as e:
        st.error(f"Gagal menghubungi SageMaker Endpoint: {e}")