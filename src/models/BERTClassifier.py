import torch
import os
from transformers import BertForSequenceClassification, BertTokenizer

class BERTClassifier:
     
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Définir le chemin du checkpoint. En fonction de l'environnement (IDE ou Docker), le chemin sera ajusté
        checkpoint_path = os.path.join(base_path, "../../models/BERT/checkpoint-750")

        # Charger le modèle BERT et le tokenizer depuis le checkpoint
        self.model = BertForSequenceClassification.from_pretrained(checkpoint_path)
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint_path)

        # Mettre le modèle en mode évaluation (désactivation des couches Dropout 
        # qui sont utilisées uniquement pendant la phase d'entrainement)
        self.model.eval()
    
    def predict(self, tweet: str):
        # Tokeniser le tweet
        inputs = self.tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Faire la prédiction
        with torch.no_grad():  # Désactiver la rétropropagation pour la prédiction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Obtenir la prédiction (classe prédite)
        predicted_class = torch.argmax(logits, dim=-1).item()
        return predicted_class

if __name__=="__main__":
    classifier = BERTClassifier()
    y_pred = classifier.predict("This movie is great!")
    print(f'Predicted class: {y_pred}')