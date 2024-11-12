from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
from dill import dump, load

class WordIndexTokenizer:

    def __init__(self, word_index=None, max_len=0):
        self.word_index = word_index
        self.max_len = max_len
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        self.sequences = None

    def fit(self, tweets):
        index = 1  # Les index commencent à 1 (généralement 0 est réservé pour le padding)
        self.word_index = defaultdict(int)

        for i, tweet in enumerate(tweets):
            if i % 100000 == 0:
                print(f"processing {i} tweets...")
            tokens = self._tokenize(tweet)
            self.max_len = max(self.max_len, len(tokens))
            if self.sequences is not None:
                self.sequences.append([])
            for token in tokens:
                if token not in self.word_index:
                    self.word_index[token] = index
                    index += 1
                if self.sequences is not None:
                    # Used in fit_transform() to avoid calling fit() and transform() separately
                    self.sequences[-1].append(self.word_index[token])

    def transform(self, tweets):
        """
        This function transforms a text into tokens and returns a list of indices where each
        index correspond to a token in the vocabulary.
        """
        sequences = []
        for tweet in tweets:
            tokens = self._tokenize(tweet)
            sequences.append([])
            for token in tokens:
                if token in self.word_index:
                    sequences[-1].append(self.word_index[token])
        return self._pad_sequences(sequences, maxlen=self.max_len)
    
    def fit_transform(self, tweets):
        # Initialize a 'sequences' attribute to avoid calling fit() and transform() separately
        self.sequences = []
        self.fit(tweets)
        res_sequences = self._pad_sequences(self.sequences, maxlen=self.max_len) 
        self.sequences = None
        
        return res_sequences

    def dump(self, sFilePath):
        """
        Save all parameters of this tokenizer into a file
        """
        data = {
            'word_index': self.word_index,
            'max_len': self.max_len
        }
        with open(sFilePath, 'wb') as f:
            dump(data, f)

    def load(self, sFilePath):
        """
        Load tokenizer parameters from a file
        """
        with open(sFilePath, 'rb') as f:
            data = load(f)
            self.word_index, self.max_len = data['word_index'], data['max_len']

    def _tokenize(self, text):
        """
        Internal function to tokenize a tweet
        """
        return [self.lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in self.stop_words]

    def _pad_sequences(self, sequences, maxlen=None, padding='pre', truncating='pre', value=0):
        """
        Pads sequences to the same length by adding a value (default 0) either at the start or the end.
        
        Parameters:
        - sequences: list of lists, where each list is a sequence of integers.
        - maxlen: int, the desired length of the output sequences. If None, the length of the longest sequence is used.
        - padding: 'pre' or 'post', where to pad the sequences (either before or after).
        - truncating: 'pre' or 'post', where to truncate the sequences (either at the beginning or at the end).
        - value: int, the value used for padding.

        Returns:
        - A 2D NumPy array with shape `(len(sequences), maxlen)`.
        """
        # Définir la longueur maximale si elle n'est pas spécifiée
        if maxlen is None:
            maxlen = max(len(s) for s in sequences)

        # Initialiser un tableau rempli de la valeur de padding
        padded_sequences = np.full((len(sequences), maxlen), value)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue  # Ignorer les séquences vides
            # Truncate the sequence if necessary
            if len(seq) > maxlen:
                if truncating == 'pre':
                    seq = seq[-maxlen:]
                else:
                    seq = seq[:maxlen]
            
            # Pad the sequence if necessary
            if padding == 'pre':
                padded_sequences[i, -len(seq):] = seq
            else:
                padded_sequences[i, :len(seq)] = seq
        
        return padded_sequences
