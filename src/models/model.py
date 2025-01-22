from sklearn.ensemble import RandomForestRegressor
import joblib

class PredictiveModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=Config.RANDOM_STATE
        )
        
    def train(self, X_train, y_train):
        """
        Train the model
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)
        
    def save_model(self, path):
        """
        Save the trained model
        """
        joblib.dump(self.model, path)
        
    def load_model(self, path):
        """
        Load a trained model
        """
        self.model = joblib.load(path)