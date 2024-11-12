import streamlit as st
import requests
import urllib.parse

AZURE_API_URL = "https://tweet-sentiment-api-brcngbash4eqafhr.westeurope-01.azurewebsites.net/predict"  # URL de l'API déployée sur Azure

# Fonction pour envoyer une requête au modèle sur Azure
def get_sentiment(text):
    # Encoder le texte pour qu'il soit sécurisé pour l'URL
    query = {"tweet": text}
    encoded_query = urllib.parse.urlencode(query)  # encode le paramètre pour l'URL
    url_with_query = f"{AZURE_API_URL}?{encoded_query}"
    response = requests.get(url_with_query)

    if response.status_code == 200:
        result = response.json()
        return result.get("predicted class", "Erreur lors de la prédiction")
    else:
        return f"Erreur de connexion à l'API: {response.status_code} - {response.json()}"

# Interface Streamlit
st.title('Analyse de Sentiment des Tweets')

# Champ de texte pour que l'utilisateur entre un tweet
tweet_text = st.text_area("Entrez votre tweet ici:")

# Bouton "Predict"
if st.button('Predict'):
    if tweet_text:
        # Appeler l'API et obtenir le sentiment
        sentiment = get_sentiment(tweet_text)
        st.write(f"Sentiment prédit : {sentiment}")
    else:
        st.warning("Veuillez entrer un tweet avant de prédire.")