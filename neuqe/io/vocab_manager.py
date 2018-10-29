class VocabManager:
    """ This is a class to manage the vocabulary. """

    def __init__(self, vocab_path, num_vocab):
        """ Initialize the vocabulary managerencoder_vocab object.

        Constructor to initialize the vocabulary manager

        Args:
            vocab_path: path to the vocabulary file, one token per line, sorted based on count (descending).
            num_vocab: number of tokens in the vocabulary to retain
        """
        # Setting the instance variables
        self.vocab_path = vocab_path
        self.vocab_size = num_vocab  + 3 # +3 for <pad>, <unk> and </s>
        self.token_to_index_dict = {}
        self.index_to_token_dict = {}

        # Setting pad symbol token
        self.pad = '<pad>'
        self.token_to_index_dict[self.pad] = 0
        self.index_to_token_dict[0] = self.pad

        # Setting end of symbol token
        self.eos = '</s>'
        self.token_to_index_dict[self.eos] = 1
        self.index_to_token_dict[0] = self.eos

        # Setting unknown token in dictionary
        self.unk = '<unk>'
        self.token_to_index_dict[self.unk] = 2
        self.index_to_token_dict[1] = self.unk


        # Loop through the vocabulary.
        index = 3
        with open(vocab_path) as fvocab:
            for line in fvocab:
                if not line:
                    continue
                token,_ = line.strip().split()
                self.token_to_index_dict[token] = index
                self.index_to_token_dict[index] = token
                index += 1
                if index >= self.vocab_size:
                    break

        # if index is less than vocab size, set it to vocab_size
        self.vocab_size = index

    def get_token(self, index):
        """Gets the token correspodning to the index.

        Args:
            index: index to the vocabulary
        Returns:
            token: the token correspondiong to the index
        """
        try:
            return self.index_to_token_dict[index]
        except KeyError:
            return self.unk

    def get_index(self, token):
        """ Gets the index corresponding to the token.

        Args:
            token: the token whose index is to be retrieved.
                    the index of the specified token or unknown if
                    token not found in vocabulary.
        """
        try:
            return self.token_to_index_dict[token]
        except KeyError:
            return self.token_to_index_dict[self.unk]
