from src.utils.WordIndexTokenizer import WordIndexTokenizer
from dill import load
from typing import List
import os

class LogisticRegressionClassifier:

    def __init__(self):
        # Calculer le chemin absolu du répertoire du projet
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Chargement du tokenizer. En fonction de l'environnement (IDE ou Docker), le chemin sera ajusté
        tokenizer_path = os.path.join(base_path, "../../tokenizer/LogisticRegression/tokenizer.pkl")
        self.tokenizer = WordIndexTokenizer()
        self.tokenizer.load(tokenizer_path)

        # Chargement du modèle de regression logistique. En fonction de l'environnement (IDE ou Docker), le chemin sera ajusté
        model_path = os.path.join(base_path, "../../models/LogisticRegression/pipeline.pkl")
        with open(model_path, 'rb') as f:
            self.pipeline = load(f)

    def predict(self, tweets:List[str]) -> List[int]:
        # Tokeniser le tweet
        sequences = self.tokenizer.transform(tweets)

        # Faire la prédiction
        y_pred = self.pipeline.predict(sequences)

        return y_pred

if __name__=="__main__":
    classifier = LogisticRegressionClassifier()
    y_pred = classifier.predict(["This movie is great!", "This movie is boring."])
    print(f'Predicted classes: {y_pred}')