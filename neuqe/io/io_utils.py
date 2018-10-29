import numpy as np
from collections import OrderedDict

def write_vocab(dataset_path, out_vocab_path):
    """Prepares vocabulary.

    Prepares the vocabulary file for a dataset. It contains all unique tokens of file with its count in descending order of token count.

    Args:
        dataset_path: path to the text file to be read from.
        out_vocab_path: path to the dictionary file to write to.

    """
    vocab = OrderedDict()
    with open(dataset_path) as f:
        lcount = 1
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
    with open(out_vocab_path,'w') as fvocab:
        for token, count in sorted(vocab.items(), key=lambda token_count: token_count[1], reverse=True):
            fvocab.write(token + ' ' + str(count) + '\n')


def create_predictor_input(batch, vocab, max_length=None):
    """Prepares predictor input

    Create numpy array for input to predictor from batch of tokens and vocabulary.

    Args:
        batch (list): list of source-target sentence pairs, each split into tokens
        vocab (tuple): tuple of source and target vocabulary.
        max_length (int): maximum number of tokens for each sentence to be included in batch.

    Returns:
        A tuple containing numpy matrices representing source, target, source_mask, and
        target_mast.
    """

    encoder_vocab, decoder_vocab = vocab

    # setting maxlen to length of longest sample
    max_length_input = max([len(sample[0]) for sample in batch])
    max_length_target = max([len(sample[1]) for sample in batch])

    # adding end of sentence marker
    inp = [[encoder_vocab.get_index(token) for token in sample[0]] +
           [encoder_vocab.get_index(encoder_vocab.eos)]
           for sample in batch]
    target = [ [decoder_vocab.get_index(decoder_vocab.eos)] + [decoder_vocab.get_index(token) for token in sample[1]] +
              [decoder_vocab.get_index(decoder_vocab.eos)] for sample in batch]
    max_length_input += 1
    max_length_target += 3 # +3 for beginning <s>, for ending <s> and also additional <pad>

    # preparing mask and input

    source_mask = np.array([[1.]*len(inp_instance[:max_length_input]) + [0.]*(max_length_input-len(inp_instance)) for inp_instance in inp], dtype='float32').transpose()
    source = np.array([inp_instance[:max_length_input]  + [0.]*(max_length_input-len(inp_instance))
                       for inp_instance in inp], dtype='int64').transpose()

    # taret preparation with -1 (beginning of sentence) row upfront

    target_mask = np.array([[1.]*len(target_instance[:max_length_target])
                            + [0.]*(max_length_target-len(target_instance))
                            for target_instance in target], dtype='float32').transpose()
    target = np.array([target_instance[:max_length_target]  +
                       [0.]*(max_length_target-len(target_instance))
                       for target_instance in target],
                      dtype='int64').transpose()
    prepared_input = (source,target,source_mask,target_mask)
    return prepared_input
