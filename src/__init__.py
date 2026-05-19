"""Project source package."""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KMeansWrapper:
    def __init__(self, kmeans, scaler):
        self.kmeans = kmeans
        self.scaler = scaler
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X.astype(float))
        return self.kmeans.predict(X_scaled)