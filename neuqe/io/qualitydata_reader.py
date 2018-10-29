import gzip
import itertools
import random

class QualityDataReader:
    def __init__(self, source_dataset_path, hypothesis_dataset_path, scores_path, batchsize=1, source_max_length=None, hypothesis_max_length=None, num_batches_in_cache=None, shuffle_batches=True):
        """
        Initialize a text iterator object to read text file batch by batch

        Method to initialize the class with the passed parameters and open the
        dataset for training.

        Args:
            source_dataset_path: path to the file containing source sentences.
            hypothesis_dataset_path

            : path to the file containing target sentences.
            batchsize: size of the minibatch.
            source_max_length: maximum length of source sentences.
            target_max_length: maximum length of target sentences.
            num_batches_in_cache: number of batches to fit in the cache
            shuffle_batches: shuffle the batches after each epoch or not

        """
        self.source_dataset_path = source_dataset_path
        self.hypothesis_dataset_path = hypothesis_dataset_path
        self.scores_path = scores_path
        self.batchsize = batchsize
        self.source_max_length = source_max_length
        self.hypothesis_max_length = hypothesis_max_length
        self.cache_size = num_batches_in_cache
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
        self.hypothesis_dataset = dataset_open(self.hypothesis_dataset_path)
        self.scores_dataset = dataset_open(self.scores_path)

        self.fill_cache()

    def reset(self):
        """
        Resets the file pointer back to the beginning
        """
        self.batch_index = 0


    def fill_cache(self):
        """ Fill the cache with batches """
        cache_samples = []
        for src_line, hyp_line, score_line in zip(self.source_dataset, self.hypothesis_dataset, self.scores_dataset):
            src_tokens = src_line.strip().split()
            hyp_tokens = hyp_line.strip().split()
            score = float(score_line.strip())
            # continue iteration if  source  max lengths
            if self.source_max_length != None and  len(src_tokens) > self.source_max_length:
                continue
            if self.hypothesis_max_length != None and len(hyp_tokens) > self.hypothesis_max_length:
                continue
            cache_samples.append((src_tokens, hyp_tokens,score))


        # sort the cache based on source size so that input to predictor is correct.
        cache_samples = sorted(cache_samples, key=lambda x: len(x[0]), reverse=True)

        self.cache = [cache_samples[i:i+self.batchsize] for i in range(0, len(cache_samples), self.batchsize)]


    def shuffle(self):
        """ Shuffles the batches in the cache """
        random.shuffle(self.cache)


    def next(self):
        """
        Default method for iterator object
        :return: returns the batch
        """
        if not self.cache:
            self.fill_cache()
            self.batch_index = 0
            print(self.cache[0], len(self.cache))
        if self.batch_index >= len(self.cache):
            self.batch_index = 0
            return None

        samples = self.cache[self.batch_index]
        self.batch_index += 1
        return samples
