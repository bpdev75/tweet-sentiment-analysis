
from src.models.LogisticRegressionClassifier import LogisticRegressionClassifier
from src.api.monitoring import Monitoring
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

API_VERSION = "1.1"

app = FastAPI()

# Chargement du model
model = LogisticRegressionClassifier()

# Chargement de l'object monitoring
monitoring = Monitoring()

# Modèle de données pour le feedback utilisateur
class Feedback(BaseModel):
    tweet: str
    predicted_sentiment: str
    feedback: bool  # True si correct, False si incorrect

@app.get("/predict")
def predict_tweet_sentiment(tweet: str):
    y_pred = model.predict([tweet])

    # Obtenir la prédiction (classe prédite)
    predicted_class = y_pred[0]
    return {"predicted class": "Positif" if predicted_class else "Négatif", "API version": API_VERSION}

@app.post("/feedback")
async def collect_feedback(feedback: Feedback):
    """
    Endpoint pour collecter le feedback utilisateur sur les prédictions.
    """
    try:
        # Loguer les informations dans Azure Application Insights
        monitoring.trace_feedback(feedback.tweet, feedback.predicted_sentiment, feedback.feedback)
        return {"message": "Feedback logged successfully"}
    except Exception as e:
        monitoring.logError("Error logging feedback")
        raise HTTPException(status_code=500, detail="Internal server error")