import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import NUM_COLS, CAT_COLS, TARGET, TARGET_MAPPING, MODEL_DIR

class DataPreprocessor:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.preprocessor = None
        
    def _convert_credit_history(self, x):
        if pd.isna(x):
            return np.nan
        x = str(x)
        y = re.search(r'(\d+)\s*Years?', x)
        m = re.search(r'(\d+)\s*Months?', x)
        years = int(y.group(1)) if y else 0
        months = int(m.group(1)) if m else 0
        return years * 12 + months

    def _clean_loan_types(self, x):
        if pd.isna(x) or str(x).lower() == 'nan':
            return np.nan
        loans = str(x).replace(' and ', ',').split(',')
        cleaned_loans = [loan.strip() for loan in loans if loan.strip()]
        return ', '.join(cleaned_loans)

    def clean_raw_data(self, df):
        """Memasukkan seluruh logic pembersihan data (Regex, Outlier, Fill NA)"""
        df_clean = df.copy()
        
        # 1. Drop unused columns
        drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Unnamed: 0']
        df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')
        
        # 2. Numeric Cleaning via Regex
        temp = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Amount_invested_monthly', 'Outstanding_Debt']
        for col in temp:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[^0-9a-zA-Z. -.]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        if 'Credit_History_Age' in df_clean.columns:
            df_clean['Credit_History_Age'] = df_clean['Credit_History_Age'].apply(self._convert_credit_history)
            df_clean['Credit_History_Age'] = pd.to_numeric(df_clean['Credit_History_Age'], errors='coerce')
            
        if 'Changed_Credit_Limit' in df_clean.columns:
            df_clean['Changed_Credit_Limit'] = df_clean['Changed_Credit_Limit'].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            df_clean['Changed_Credit_Limit'] = pd.to_numeric(df_clean['Changed_Credit_Limit'], errors='coerce')

        # 3. Rules / Outlier Filtering (Set to NaN)
        if 'Age' in df_clean.columns:
            df_clean.loc[(df_clean['Age'] < 18) | (df_clean['Age'] > 100), 'Age'] = np.nan
        if 'Num_of_Loan' in df_clean.columns:
            df_clean.loc[(df_clean['Num_of_Loan'] < 0) | (df_clean['Num_of_Loan'] > 100), 'Num_of_Loan'] = np.nan
        if 'Num_Bank_Accounts' in df_clean.columns:
            df_clean.loc[(df_clean['Num_Bank_Accounts'] < 0) | (df_clean['Num_Bank_Accounts'] > 100), 'Num_Bank_Accounts'] = np.nan
        if 'Interest_Rate' in df_clean.columns:
            df_clean.loc[(df_clean['Interest_Rate'] < 0) | (df_clean['Interest_Rate'] > 100), 'Interest_Rate'] = np.nan
        if 'Num_Credit_Card' in df_clean.columns:
            df_clean.loc[df_clean['Num_Credit_Card'] > 20, 'Num_Credit_Card'] = np.nan
        if 'Num_Credit_Inquiries' in df_clean.columns:
            df_clean.loc[df_clean['Num_Credit_Inquiries'] > 50, 'Num_Credit_Inquiries'] = np.nan
        if 'Delay_from_due_date' in df_clean.columns:
            df_clean.loc[df_clean['Delay_from_due_date'] < 0, 'Delay_from_due_date'] = np.nan
        if 'Num_of_Delayed_Payment' in df_clean.columns:
            df_clean.loc[(df_clean['Num_of_Delayed_Payment'] < 0) | (df_clean['Num_of_Delayed_Payment'] > 100), 'Num_of_Delayed_Payment'] = np.nan
        if 'Changed_Credit_Limit' in df_clean.columns:
            df_clean.loc[df_clean['Changed_Credit_Limit'] < 0, 'Changed_Credit_Limit'] = np.nan

        # 4. Fill NA for Numeric Columns (with Median)
        for col in NUM_COLS:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                # Jika seluruh baris NaN (contoh input 1 baris di Streamlit), isi dengan 0
                if pd.isna(median_val):
                    median_val = 0  
                df_clean[col].fillna(median_val, inplace=True)

        # 5. Specific Categorical Cleaning
        if 'Occupation' in df_clean.columns:
            df_clean['Occupation'] = df_clean['Occupation'].str.replace('_', '', regex=False).replace('', np.nan)
        if 'Credit_Mix' in df_clean.columns:
            df_clean['Credit_Mix'] = df_clean['Credit_Mix'].str.replace('_', '', regex=False).replace('', np.nan)
        if 'Type_of_Loan' in df_clean.columns:
            df_clean['Type_of_Loan'] = df_clean['Type_of_Loan'].apply(self._clean_loan_types).replace(['nan', 'None', ''], np.nan)
        if 'Payment_Behaviour' in df_clean.columns:
            df_clean['Payment_Behaviour'] = df_clean['Payment_Behaviour'].astype(str).str.replace(r'[^a-zA-Z_]', '', regex=True).replace('', np.nan)

        # 6. Fill NA for Categorical Columns
        if 'Occupation' in df_clean.columns:
            df_clean['Occupation'] = df_clean['Occupation'].fillna('Unknown')
        if 'Credit_Mix' in df_clean.columns:
            df_clean['Credit_Mix'] = df_clean['Credit_Mix'].fillna('Unknown')
        if 'Payment_Behaviour' in df_clean.columns:
            df_clean['Payment_Behaviour'] = df_clean['Payment_Behaviour'].fillna('Unknown')
        if 'Payment_of_Min_Amount' in df_clean.columns:
            df_clean['Payment_of_Min_Amount'] = df_clean['Payment_of_Min_Amount'].replace('NM', 'Unknown')
            df_clean['Payment_of_Min_Amount'] = df_clean['Payment_of_Min_Amount'].fillna('Unknown')
        if 'Type_of_Loan' in df_clean.columns:
            df_clean['Type_of_Loan'] = df_clean['Type_of_Loan'].fillna('No_Loan')

        # 7. Final Categorical Regex Check
        for col in CAT_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[^0-9a-zA-Z._ ,-]', '', regex=True)

        return df_clean

    def prepare_training_data(self, df):
        df = self.clean_raw_data(df)
        
        # Target mapping
        df[TARGET] = df[TARGET].map(TARGET_MAPPING)
        
        # MultiLabelBinarizer untuk Type_of_Loan
        loan_list = df['Type_of_Loan'].apply(lambda x: [i.strip() for i in str(x).split(',')] if x != 'No_Loan' else [])
        loan_encoded = self.mlb.fit_transform(loan_list)
        df_loan_results = pd.DataFrame(loan_encoded, columns=self.mlb.classes_, index=df.index)
        
        df = pd.concat([df, df_loan_results], axis=1).drop(columns=['Type_of_Loan'])
        
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        
        # ColumnTransformer: Scale (Numerik) dan OneHotEncode (Kategorikal pengganti get_dummies)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUM_COLS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
            ],
            remainder='passthrough'
        )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Transform 
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Save Artifacts
        joblib.dump(self.preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
        joblib.dump(self.mlb, f"{MODEL_DIR}/mlb.pkl")
        
        return X_train_processed, X_test_processed, y_train, y_test