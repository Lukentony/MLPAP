class Config:
    # Parametri base
    DATA_DIR = "raw/"
    MODEL_PATH = "models/saved/"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Parametri dataset
    FILENAME = "covid19_italy_province.csv"  # Nome del file da processare
    TARGET_COLUMN = "TotalPositiveCases"     # Colonna target
    DATE_COLUMNS = ["Date"]                  # Colonne data
    CATEGORICAL_COLUMNS = ["RegionName", "ProvinceName"]  # Colonne categoriche
    DROP_COLUMNS = []                        # Colonne da escludere