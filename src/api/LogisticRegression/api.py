
from src.models.LogisticRegressionClassifier import LogisticRegressionClassifier
from src.api.monitoring import Monitoring
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

API_VERSION = "1.1"

app = FastAPI()

# Chargement du model
model = LogisticRegressionClassifier()

# Modèle de données pour le feedback utilisateur
class Feedback(BaseModel):
    tweet: str
    predicted_sentiment: str
    feedback: bool  # True si correct, False si incorrect

# Fonction de dépendance pour initialiser Monitoring
def get_monitoring():
    return Monitoring()

@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    # Initialisation manuelle de Monitoring ici (car Depends() ne fonctionne pas dans un gestionnaire global)
    monitoring = Monitoring()

    monitoring.logError(str(exc))
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "error": str(exc)}
    )

@app.get("/predict")
def predict_tweet_sentiment(tweet: str):
    y_pred = model.predict([tweet])

    # Obtenir la prédiction (classe prédite)
    predicted_class = y_pred[0]
    return {"predicted class": "Positif" if predicted_class else "Négatif", "API version": API_VERSION}

@app.post("/feedback")
async def collect_feedback(feedback: Feedback, monitoring: Monitoring = Depends(get_monitoring)):
    """
    Endpoint pour collecter le feedback utilisateur sur les prédictions.
    """
    try:
        # Loguer les informations dans Azure Application Insights
        monitoring.trace_feedback(feedback.tweet, feedback.predicted_sentiment, feedback.feedback)
        return {"message": "Feedback logged successfully"}
    except Exception as e:
        monitoring.logError(f"Error logging feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "exception": str(e)
            }
        )