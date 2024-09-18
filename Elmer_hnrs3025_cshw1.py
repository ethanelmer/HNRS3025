import argparse
import pickle
import random
import re
from collections import defaultdict, Counter

class NgramModel:
    def __init__(self, n):
        """
        Initialize the n-gram model.
        Args:
            n (int): The order of the n-gram (1 for unigram, 2 for bigram).
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def preprocess(self, text):
        """
        Convert text to lowercase and tokenize, treating punctuation as separate tokens.
        Args:
            text (str): The input text.
        Returns:
            tokens (list): A list of tokens.
        """
        lower_text = text.lower()  #Converts text to lowercase
        tokens = re.findall(r"\b\w+", lower_text)  #\b\w+ = cuts out all whitespace and punctuation
        return tokens

    def train(self, text_corpus):
        """
        Train the n-gram model on the provided text corpus.
        Args:
            text_corpus (str): The input text corpus.
        """
        tokens = self.preprocess(text_corpus)  #Preprocesses the text (using the method above)
        self.vocab.update(tokens)  #Updates the vocab with the new tokens
        for i in range(self.n, len(tokens)):
            context = tuple(tokens[i - self.n:i])  #Assigns the context to the current tuple
            word = tokens[i]  #Moves to the current word
            self.ngram_counts[context][word] += 1
            self.context_counts[context] += 1

    def predict_next_word(self, context, deterministic=False):
        """
        Predict the next word based on the context.
        Args:
            context (tuple): Tuple containing the prior word(s).
            deterministic (bool): If True, always select the highest probability word.
        Returns:
            str: The predicted next word, or None if an error occurs.
        """
        for word in context:
            if word not in self.vocab:
                print(f"Error: Word '{word}' not found in the vocabulary.")  #Prints an error message if the word is not in the training set
                return None
        word_counts = self.ngram_counts[context]  #Count of an individual word
        total_counts = self.context_counts[context]  #Count of all words in the corpus
        if deterministic:
            next_word = word_counts.most_common(1)[0][0]  #Next word is the word with the highest probability
        else:
            words, counts = zip(*word_counts.items())  #Extracts the words and their corresponding counts and separates them into tuples
            prob_calc = [count / total_counts for count in counts]  #Calculates the probability of each word
            next_word = random.choices(words, weights=prob_calc, k=1)[0]  #Picks words based on their probabilities
        return next_word

    def save(self, filepath):
        """
        Save the n-gram model to a pickle file.
        Args:
            filepath (str): The output file path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)  #Dump the contents of the training into the filepath

    @staticmethod
    def load(filepath):
        """
        Load the n-gram model from a pickle file.
        Args:
            filepath (str): The output file path.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)  #Load the contents of the training data from the file path

class BPEModel:
    """A class for the Byte Pair Encoding (BPE) tokenizer."""
    def __init__(self):
        self.vocabulary = set()
        self.token_to_id = {}
        self.id_to_token = {}

    def preprocess(self, text):
        """Preprocess text by lowercasing and splitting into words.
        Args:
            text (str): The text to preprocess.
        Returns:
            list: A list of words.
        """
        lower_text = text.lower()
        words = re.findall(r'\b\w+|\b[^\w\s]', lower_text)  #\b\w+|\b[^\w\s] = cuts out all whitespace and leaves punctuation
        return words

    def get_stats(self, tokens):
        """Compute frequencies of pairs of symbols.
        Args:
            tokens (dict): Dictionary of tokenized words and their frequencies.
        Returns:
            dict: Frequencies of symbol pairs.
        """
        pairs = defaultdict(int)
        for word, freq in tokens.items():
            symbols = word.split()  #Splits the symbols from the words
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq  #Pairs the symbols with their frequencies
        return pairs

    def merge_vocab(self, pair, tokens):
        """Merge the most frequent pair in all tokenized words.
        Args:
            pair (tuple): The symbol pair to merge.
            tokens (dict): Current tokens and their frequencies.
        Returns:
            dict: Updated tokens after merging.
        """
        bigram = ' '.join(pair)  #Adds the pair to the bigram
        replacement = ''.join(pair)  #Adds the pair to the replacement
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')  #Matches the bigram when it is a whole token
        new_tokens = {}
        for word in tokens:
            new_word = pattern.sub(replacement, word)  #Sets the new word
            new_tokens[new_word] = tokens[word]  #Sets the new token
        return new_tokens

    def train(self, data_string, k):
        """
        Train the BPE model using the provided data string.
        Args:
            data_string (str): The training data corpus.
            k (int, optional): Number of BPE merge operations. Defaults to 500.
        """
        words = self.preprocess(data_string)  #Preprocesses using the method above
        token_freqs = Counter(words)
        tokens = {' '.join(list(word) + ['</w>']): freq for word, freq in token_freqs.items()}  #Puts each token next to the end word symbol
        for _ in range(k):
            pairs = self.get_stats(tokens)  #Fetches the pairs
            if not pairs:
                break
            most_common = max(pairs, key=pairs.get)  #Gets the pair that appears the most
            tokens = self.merge_vocab(most_common, tokens)
        self.vocabulary = set()
        for word in tokens:
            self.vocabulary.update(word.split())  #Updates the vocabulary
        sorted_tokens = sorted(self.vocabulary)
        self.token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def tokenize(self, text_string):
        """Tokenize the provided string using the trained BPE vocabulary.
        Args:
            text_string (str): The text to tokenize.
        Returns:
            tuple: A tuple containing the tokens and their corresponding IDs.
        """
        if not self.vocabulary:
            print("Error: BPE model is not trained yet.")
            return None, None
        words = self.preprocess(text_string)
        tokens = []
        token_ids = []
        for word in words:
            word_tokens = self.bpe_encode(word)
            tokens.extend(word_tokens)
            token_ids.extend([self.token_to_id.get(token, -1) for token in word_tokens])  #Convertes BPE tokens into numerical IDs
        return tokens, token_ids

    def bpe_encode(self, word):
        """Encode a single word using the BPE vocabulary.
        Args:
            word (str): The word to encode.
        Returns:
            list: A list of BPE tokens representing the word.
        """
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            candidate_pairs = {pair for pair in pairs if ''.join(pair) in self.vocabulary}
            if not candidate_pairs:
                break
            most_frequent_pair = max(candidate_pairs, key=lambda p: self.vocabulary)  #Finds the most frequent pair
            i = 0
            while i < len(word) - 1:
                if word[i] == most_frequent_pair[0] and word[i + 1] == most_frequent_pair[1]:
                    word[i] = ''.join(most_frequent_pair) #add word to the most frequent pair
                    del word[i + 1]
                else:
                    i += 1
        if word[-1] == '</w>':
            word = word[:-1]
        return word

    def save(self, filepath):
        """Save the BPE model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a BPE model from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description='N-gram Model and BPE Tokenizer')
    parser.add_argument('activity', choices=['train_ngram', 'predict_ngram', 'train_bpe', 'tokenize'], help='Select activity to perform.')
    parser.add_argument('--data', type=str, help='Path to the training data corpus.')
    parser.add_argument('--save', type=str, help='Path to save the trained model.')
    parser.add_argument('--load', type=str, help='Path to load the trained model.')
    parser.add_argument('--word', type=str, help='Starting word(s) for prediction (use space to separate words).')
    parser.add_argument('--nwords', type=int, help='Number of words to predict.')
    parser.add_argument('--text', type=str, help='Text to tokenize.')
    parser.add_argument('--n', type=int, choices=[1, 2], help='Order of the n-gram (1 or 2).')
    parser.add_argument('--d', action='store_true', help='Deterministic prediction flag.')
    parser.add_argument('--k', type = int, help='Determines the number of iterations to run for the BPE model to run.')

    args = parser.parse_args()

    #Train n-gram handling
    if args.activity == 'train_ngram':
        if not args.data or not args.save or args.n is None:
            print("Error: --data, --save, and --n are required for train_ngram.")
            return
        with open(args.data, 'r', encoding='utf-8') as f:
            data_string = f.read()
        model = NgramModel(n=args.n)
        model.train(data_string)  #Train the model
        model.save(args.save)  #Save the model
        print(f"N-gram model (n={args.n}) trained and saved to {args.save}.")

    # Predict n-gram handling
    elif args.activity == 'predict_ngram':
        if not args.load or not args.word or args.nwords is None:
            print("Error: --load, --word, and --nwords are required for predict_unigram.")
            return

        #Unigram model
        if args.n == 1:
            model = NgramModel.load(args.load)
            if model.n != 1:
                print("Error: Loaded model is not a unigram model.")
                return
            input_words = tuple(args.word.lower().split())  #Create the input words
            if len(input_words) != 1:
                print("Error: Unigram model requires exactly one starting word.")
                return
            context = input_words
            generated_words = list(input_words)
            for _ in range(args.nwords):
                next_word = model.predict_next_word(context, deterministic=args.d)
                if next_word is None:
                    break
                generated_words.append(next_word)
                context = (next_word.lower(),)
            if(len(generated_words) > 1):
                print(' '.join(generated_words))

        #Bigram model
        else:
            model = NgramModel.load(args.load)
            if model.n != 2:
                print("Error: Loaded model is not a bigram model.")
                return
            input_words = tuple(args.word.lower().split())
            if len(input_words) != 2:
                print("Error: Bigram model requires exactly two starting words.")
                return
            context = input_words
            generated_words = list(input_words)
            for _ in range(args.nwords):
                next_word = model.predict_next_word(context, deterministic=args.d)
                if next_word is None:
                    break
                generated_words.append(next_word)
                context = (context[1], next_word.lower())
            if len(generated_words) > 2:
                print(' '.join(generated_words))

    #Train BPE handling
    elif args.activity == 'train_bpe':
        if not args.data or not args.save:
            print("Error: --data and --save are required for train_bpe.")
            return
        with open(args.data, 'r', encoding='utf-8') as f:
            data_string = f.read()
        model = BPEModel()

        #Use user defined k
        if(args.k):
            model.train(data_string, args.k)

        #Use default k
        else:
            model.train(data_string, 500)
        model.save(args.save)
        print(f"BPE model trained and saved to {args.save}.")

    #Tokenize handling
    elif args.activity == 'tokenize':
        if not args.load or not args.text:
            print("Error: --load and --text are required for tokenize.")
            return
        model = BPEModel.load(args.load)
        tokens, token_ids = model.tokenize(args.text)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
