import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
    def preprocess(self, data):
        df = data.copy()
        
        # Separa colonne numeriche e categoriche
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = [col for col in Config.CATEGORICAL_COLUMNS if col in df.columns]
        
        # Gestione valori mancanti
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.numerical_imputer.fit_transform(df[numeric_cols])
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        # Encoding categoriche
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Scaling numeriche
        for col in numeric_cols:
            self.scalers[col] = StandardScaler()
            df[col] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1))
        
        return df
