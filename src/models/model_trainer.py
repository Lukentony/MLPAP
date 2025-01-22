from sklearn.metrics import mean_squared_error, r2_score
import logging

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train the model and evaluate its performance
        """
        # Train the model
        self.model.train(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log results
        self.logger.info(f"Mean Squared Error: {mse}")
        self.logger.info(f"R2 Score: {r2}")
        
        return mse, r2
