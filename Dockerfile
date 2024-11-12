# Utilise l'image Python 3.11.10 comme base
FROM python:3.11-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installe les dépendances à partir de requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Télécharge les stopwords
RUN python -m nltk.downloader stopwords punkt punkt_tab wordnet

# Copier les répertoires src et models dans le répertoire de travail /app
COPY src/ src/
COPY models/ models/
COPY tokenizer/ tokenizer/

EXPOSE 8000

# Commande pour exécuter l'application
CMD ["uvicorn", "src.api.LogisticRegression.api:app", "--host", "0.0.0.0", "--port", "8000"]