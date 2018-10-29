import gzip
import itertools
import random

class ParallelReader:
    """
    Class for reading parallel data

    TODO: caching not implemented. currently loads the entire data to memory.
    """
    def __init__(self, source_dataset_path, target_dataset_path, batchsize=1, source_max_length=None, target_max_length=None, num_batches_in_cache=None, shuffle_batches=True):
        """
        Initialize a text iterator object to read text file batch by batch.

        Method to initialize the class with the passed parameters and open the
        dataset for training.

        Args:
            source_dataset_path: path to the file containing source sentences.
            target_dataset_path: path to the file containing target sentences.
            batchsize: size of the minibatch.
            source_max_length: maximum length of source sentences.
            target_max_length: maximum length of target sentences.
            num_batches_in_cache: number of batches to fit in the cache
            shuffle_batches: shuffle the batches after each epoch or not

        """
        self.source_dataset_path = source_dataset_path
        self.target_dataset_path = target_dataset_path
        self.batchsize = batchsize
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.cache_size = num_batches_in_cache # TODO: not implemented
        self.cache = []
        self.shuffle_batches = shuffle_batches

        # Opining dataset
        def dataset_open(dataset_path):
            if dataset_path.endswith('.gz'):
                dataset = gzip.open(dataset_path,'r')
            else:
                dataset = open(dataset_path, 'r')
            return dataset

        self.source_dataset = dataset_open(self.source_dataset_path)
        self.target_dataset = dataset_open(self.target_dataset_path)

        self.fill_cache()
        self.batch_index = 0



    def reset(self):
        """ Resets the file pointer back to the beginning """
        self.batch_index = 0


    def fill_cache(self):
        """
        Fills the cache upto its size.
        """
        cache_samples = []
        for src_line, trg_line in zip(self.source_dataset, self.target_dataset):
            src_tokens = src_line.strip().split()
            trg_tokens = trg_line.strip().split()
            # continue iteration if  source  max lengths
            if self.source_max_length != None and  len(src_tokens) > self.source_max_length:
                continue
            if self.target_max_length != None and len(trg_tokens) > self.target_max_length:
                continue
            cache_samples.append((src_tokens, trg_tokens))


        # sort the cache based on target size
        cache_samples = sorted(cache_samples, key=lambda x: len(x[0]), reverse=True)
        self.cache = [cache_samples[i:i+self.batchsize] for i in range(0, len(cache_samples), self.batchsize)]

        self.shuffle()

    def shuffle(self):
        """ Shuffle the batches. """

        random.shuffle(self.cache)

    def next(self):
        """
        Default method for iterator object

        Returns:
            One minibatch (a list of samples). A sample consists of source and target tokens.
        """
        if not self.cache:
            self.fill_cache()
            self.batch_index = 0
        if self.batch_index >= len(self.cache):
            self.batch_index = 0
            return None

        samples = self.cache[self.batch_index]
        self.batch_index += 1
        return samples
