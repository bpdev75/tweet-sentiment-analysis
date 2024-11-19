import streamlit as st
import requests
import urllib.parse

AZURE_BASE_URL = "https://tweet-sentiment-api-brcngbash4eqafhr.westeurope-01.azurewebsites.net" # URL de l'API déployée sur Azure
AZURE_PREDICT_URL = f"{AZURE_BASE_URL}/predict"
AZURE_FEEDBACK_URL = f"{AZURE_BASE_URL}/feedback"

# Fonction pour envoyer une requête au modèle sur Azure
def get_sentiment(text):
    # Encoder le texte pour qu'il soit sécurisé pour l'URL
    query = {"tweet": text}
    encoded_query = urllib.parse.urlencode(query)  # encode le paramètre pour l'URL
    url_with_query = f"{AZURE_PREDICT_URL}?{encoded_query}"
    response = requests.get(url_with_query)

    if response.status_code == 200:
        result = response.json()
        return result.get("predicted class", "Erreur lors de la prédiction")
    else:
        return f"Erreur de connexion à l'API: {response.status_code} - {response.json()}"

def send_feedback(tweet, predicted_sentiment, feedback):
    data = {
        "tweet": tweet,
        "predicted_sentiment": predicted_sentiment,
        "feedback": feedback
    }
    response = requests.post(AZURE_FEEDBACK_URL, json=data)
    if response.status_code == 200:
        st.success("Feedback envoyé avec succès !")
    else:
        st.error(f"Erreur lors de l'envoi du feedback : {response.text}")

# Interface Streamlit
st.title('Analyse de Sentiment des Tweets')

# Champ de texte pour que l'utilisateur entre un tweet
tweet_text = st.text_area("Entrez votre tweet ici:")

# Bouton "Predict"
if st.button('Predict'):
    if tweet_text:
        # Enregistrer l'état du bouton "Predict" dans st.session_state
        st.session_state.predict_clicked = True
    else:
        st.warning("Veuillez entrer un tweet avant de prédire.")

# Afficher les boutons "Correct" et "Incorrect" seulement après avoir cliqué sur "Predict"
if 'predict_clicked' in st.session_state and st.session_state.predict_clicked:
    # Appeler l'API et obtenir le sentiment
    sentiment = get_sentiment(tweet_text)
    st.write(f"Sentiment prédit : {sentiment}")

    # Ajouter les boutons de feedback
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('👍 Correct'):
            st.info('Merci pour votre feedback', icon="ℹ️")
    with col2:
        if st.button('👎 Incorrect'):
            send_feedback(tweet_text, sentiment, False)