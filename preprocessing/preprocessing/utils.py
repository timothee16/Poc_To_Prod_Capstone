import numpy as np
import pandas as pd

from keras.utils import Sequence
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


def integer_floor(float_value: float):
    """
    link to doc for numpy.floor https://numpy.org/doc/stable/reference/generated/numpy.floor.html
    """
    return int(np.floor(float_value))


class _SimpleSequence(Sequence):
    """
    Base object for fitting to a sequence of data, such as a dataset.
    link to doc : https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, get_batch_method, num_batches_method):
        self.get_batch_method = get_batch_method
        self.num_batches_method = num_batches_method

    def __len__(self):
        return self.num_batches_method()

    def __getitem__(self, idx):
        return self.get_batch_method()


class BaseTextCategorizationDataset:
    """
    Generic class for text categorization
    data sequence generation
    """

    def __init__(self, batch_size, train_ratio=0.8):
        assert train_ratio < 1.0
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def _get_label_list(self):
        """
        returns list of labels
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_num_labels(self):
        """
        returns the number of labels
        """
        # TODO: CODE HERE
        return len(self._get_label_list())

    def _get_num_samples(self):
        """
        returns number of samples (dataset size)
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def _get_num_train_samples(self):
        """
        returns number of train samples
        (training set size)
        """
        # TODO: CODE HERE
        #print("num samples",self._get_num_samples())
        #print('train ratio',self.train_ratio)
        return integer_floor(self._get_num_samples()*self.train_ratio)

    def _get_num_test_samples(self):
        """
        returns number of test samples
        (test set size)
        """
        # TODO: CODE HERE
        return self._get_num_samples()-self._get_num_train_samples()

    def _get_num_train_batches(self):
        """
        returns number of train batches
        """
        # TODO: CODE HERE
        #print("num train samples",self._get_num_train_samples())
        #print("batch size",self.batch_size)
        return integer_floor(self._get_num_train_samples()/self.batch_size)


    def _get_num_test_batches(self):
        """
        returns number of test batches
        """
        # TODO: CODE HERE
        return integer_floor(self._get_num_test_samples()/self.batch_size)

    def get_train_batch(self):
        """
        returns next train batch
        should not be implemented in this class (we can assume it's a given)
        """
        raise NotImplementedError

    def get_test_batch(self):
        """
        returns next test batch
        should not be implemented in this class (we can assume it's a given)
        """
        raise NotImplementedError

    def get_index_to_label_map(self):
        """
        from label list, returns a map index -> label
        (dictionary index: label)
        """
        # TODO: CODE HERE
        recto=self._get_label_list()
        verso=self.get_num_labels()
        verso = [i for i in range(verso)]
        return dict(zip(recto, verso))

    def get_label_to_index_map(self):
        """
        from index -> label map, returns label -> index map
        (reverse the previous dictionary)
        """
        # TODO: CODE HERE
        recto = self.get_num_labels()
        verso = self._get_label_list()
        verso = [i for i in range(verso)]
        return dict(zip(recto, verso))

    def to_indexes(self, labels):
        """
        from a list of labels, returns a list of indexes
        """
        # TODO: CODE HERE
        #labels=self.labels
        answer = []
        keys=self.get_index_to_label_map()
        for label in labels:
            value = keys.get(label)
            answer.append(value)
        return answer

    def get_train_sequence(self):
        """
        returns a train sequence of type _SimpleSequence
        """
        return _SimpleSequence(self.get_train_batch, self._get_num_train_batches)

    def get_test_sequence(self):
        """
        returns a test sequence of type _SimpleSequence
        """
        # TODO: CODE HERE
        return _SimpleSequence(self.get_test_batch, self._get_num_test_batches)

    def __repr__(self):
        return self.__class__.__name__ + \
               f"(n_train_samples: {self._get_num_train_samples()}, " \
               f"n_test_samples: {self._get_num_test_samples()}, " \
               f"n_labels: {self.get_num_labels()})"

class LocalTextCategorizationDataset(BaseTextCategorizationDataset):
    """
    A TextCategorizationDataset read from a file residing in the local filesystem
    """

    def __init__(self, filename, batch_size,
                 train_ratio=0.8, min_samples_per_label=100, preprocess_text=lambda x: x):
        """
        :param filename: a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
        :param batch_size: number of samples per batch
        :param train_ratio: ratio of samples dedicated to training set between (0, 1)
        :param preprocess_text: function taking an array of text and returning a numpy array, default identity
        """
        super().__init__(batch_size, train_ratio)
        self.filename = filename
        self.preprocess_text = preprocess_text

        self._dataset = self.load_dataset(filename, min_samples_per_label)

        #print(self._get_num_train_batches())
        assert self._get_num_train_batches() > 0
        assert self._get_num_test_batches() > 0

        # TODO: CODE HERE
        # from self._dataset, compute the label list
        self._label_list =self._get_label_list()

        y = self.to_indexes(self._dataset['tag_name'])
        y = to_categorical(y, num_classes=len(self._label_list))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._dataset['title'],
            y,
            train_size=self._get_num_train_samples(),
            stratify=y)

        self.train_batch_index = 0
        self.test_batch_index = 0

    @staticmethod
    def load_dataset(filename, min_samples_per_label):
        """
        loads dataset from filename apply pre-processing steps (keeps only tag_position = 0 & removes tags that were
        seen less than `min_samples_per_label` times)
        """

        # reading dataset from filename path, dataset is csv
        # TODO: CODE HERE
        df = pd.read_csv(filename)
        # assert that columns are the ones expected
        # TODO: CODE HERE
        assert ((df.columns == ['post_id', 'tag_name', 'tag_id', 'tag_position', 'title']).all())

        def filter_tag_position(position):
            def filter_function(df):
                """
                keep only tag_position = position
                """
                # TODO: CODE HERE
                return df[df["tag_position"]==position]

            return filter_function

        def filter_tags_with_less_than_x_samples(x):
            def filter_function(df):
                """
                removes tags that are seen less than x times
                """
                # TODO: CODE HERE
                filtered_df = (df.tag_id.value_counts() >= x)
                tag = filtered_df[filtered_df == True].index
                return df.loc[df['tag_id'].isin(tag)]

            return filter_function

        # use pandas.DataFrame.pipe to chain preprocessing steps
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pipe.html
        # return pre-processed dataset
        # TODO: CODE HERE
        df_processed = (df.
                        pipe(filter_tag_position(0)).
                        pipe(filter_tags_with_less_than_x_samples(min_samples_per_label))
                        )
        return df_processed.reset_index()

    # we need to implement the methods that are not implemented in the super class BaseTextCategorizationDataset

    def _get_label_list(self):
        """
        returns label list
        """
        # TODO: CODE HERE
        return self._dataset.tag_name.unique()

    def _get_num_samples(self):
        """
        returns number of samples in dataset
        """
        # TODO: CODE HERE
        #print("pipo",self._dataset)
        #print("lala",len(self._dataset))
        return len(self._dataset)

    def get_train_batch(self):
        # TODO: CODE HERE
        i = self.train_batch_index
        # takes x_train between i * batch_size to (i + 1) * batch_size, and apply preprocess_text
        next_x = self.preprocess_text(self.x_train.iloc[i*self.batch_size:(i + 1)*self.batch_size])
        # TODO: CODE HERE
        # takes y_train between i * batch_size to (i + 1) * batch_size
        next_y =self.y_train[i*self.batch_size:(i + 1)*self.batch_size]
        # When we reach the max num batches, we start anew
        self.train_batch_index = (self.train_batch_index + 1) % self._get_num_train_batches()
        return next_x,next_y



    def get_test_batch(self):
        """
        it does the same as get_train_batch for the test set
        """
        # TODO: CODE HERE
        i = self.test_batch_index
        # takes x_train between i * batch_size to (i + 1) * batch_size, and apply preprocess_text
        next_x = self.preprocess_text(self.x_test.iloc[i*self.batch_size:(i + 1) * self.batch_size])
        # TODO: CODE HERE
        # takes y_train between i * batch_size to (i + 1) * batch_size
        next_y = self.y_test[i*self.batch_size:(i + 1) * self.batch_size]
        # When we reach the max num batches, we start anew
        self.test_batch_index = (self.test_batch_index + 1) % self._get_num_test_batches()
        return next_x, next_y