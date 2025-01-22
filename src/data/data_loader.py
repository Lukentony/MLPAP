import pandas as pd
from sklearn.model_selection import train_test_split
import os
from config import Config

class DataLoader:
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        
    def get_available_datasets(self):
        """Lista tutti i dataset disponibili nella directory raw"""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
    
    def load_data(self, filename=None):
        """Carica il dataset specificato o quello di default"""
        if filename is None:
            filename = Config.FILENAME
            
        file_path = os.path.join(self.data_dir, filename)
        data = pd.read_csv(file_path)
        
        # Gestione date
        for date_col in Config.DATE_COLUMNS:
            if date_col in data.columns:
                data[date_col] = pd.to_datetime(data[date_col])
                data[f'{date_col}_Day'] = data[date_col].dt.day
                data[f'{date_col}_Month'] = data[date_col].dt.month
                data[f'{date_col}_Year'] = data[date_col].dt.year
                
        return data

    def prepare_data(self, data):
        """Prepara X e y dal dataset"""
        # Rimuovi colonne specificate
        drop_cols = Config.DROP_COLUMNS + Config.DATE_COLUMNS
        if Config.TARGET_COLUMN in data.columns:
            X = data.drop([Config.TARGET_COLUMN] + drop_cols, axis=1)
            y = data[Config.TARGET_COLUMN]
            return X, y
        else:
            raise ValueError(f"Target column {Config.TARGET_COLUMN} not found in dataset")

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=Config.TEST_SIZE, 
                              random_state=Config.RANDOM_STATE)