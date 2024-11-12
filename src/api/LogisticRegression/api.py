
from src.models.LogisticRegressionClassifier import LogisticRegressionClassifier
from fastapi import FastAPI

app = FastAPI()

# Chargement du model
model = LogisticRegressionClassifier()

@app.get("/predict")
def predict_tweet_sentiment(tweet: str):
    y_pred = model.predict([tweet])

    # Obtenir la prédiction (classe prédite)
    predicted_class = y_pred[0]
    return {"predicted class": "Positif" if predicted_class else "Négatif"}