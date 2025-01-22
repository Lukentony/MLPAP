import os  # Per os.path.join
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.predictive_model import PredictiveModel 
from src.model_trainer import ModelTrainer
from src.utils.logger import setup_logger
from src.config import Config

def main():
    logger = setup_logger()
    
    try:
        # Inizializza componenti
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        model = PredictiveModel()
        trainer = ModelTrainer(model)
        
        # Lista dataset disponibili
        available_datasets = data_loader.get_available_datasets()
        logger.info(f"Dataset disponibili: {available_datasets}")
        
        # Carica dati
        data = data_loader.load_data()
        
        # Prepara feature e target
        X, y = data_loader.prepare_data(data)
        
        # Preprocess
        X_processed = preprocessor.preprocess(X)
        
        # Split e training
        X_train, X_test, y_train, y_test = data_loader.split_data(X_processed, y)
        mse, r2 = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Salva
        model_filename = f"model_{Config.FILENAME.split('.')[0]}.joblib"
        model.save_model(os.path.join(Config.MODEL_PATH, model_filename))
        
        logger.info(f"Training completato. MSE: {mse:.4f}, R2: {r2:.4f}")
        
    except Exception as e:
        logger.error(f"Errore: {str(e)}")
        raise

if __name__ == "__main__":
    main()