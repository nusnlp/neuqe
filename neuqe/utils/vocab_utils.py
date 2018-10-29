from collections import Counter

def write_vocab(dataset_path, out_vocab_path):
    """ Writes vocabular to file
    Prepares the vocabulary file for a dataset. It contains all unique tokens of file with its count, in descending order of token count.

    Args:
        dataset_path: path to the text file to be read from.
        out_vocab_path: path to the dictionary file to write to.
    """
    ctr = Counter()
    with open(dataset_path) as f:
        lcount = 1
        for line in f:
            tokens = line.strip().split()
            ctr.update(tokens)
    with open(out_vocab_path,'w') as fvocab:
        for token, count in ctr.most_common():
            fvocab.write(token + ' ' + str(count) + '\n')