import cPickle
import theano
import numpy
import csv
import sys
import logging

class DataSet:
    """
    Base class for all types of datasets.
    All dataset classes should have following data members:
    tr_x, tr_y
    va_x, va_y
    te_x, te_y
    Also:
    np_tr_x, np_tr_y
    np_va_x, np_va_y
    np_te_x, np_te_y
    Note that not-pre-fixed variables should be theano.shared
    and np_ pre-fixed variables should be numpy.2D-array
    """
    def get_number_of_batches(self, batch_size):
        n_tr_batches = self.np_tr_x.shape[0] / batch_size
        n_va_batches = self.np_va_x.shape[0] / batch_size
        n_te_batches = self.np_te_x.shape[0] / batch_size

        return n_tr_batches, n_va_batches, n_te_batches

    def make_it_shared(is_int=False):
        self.tr_y = DataSet.get_shared(self.np_tr_y, is_int=is_int)
        self.va_y = DataSet.get_shared(self.np_va_y, is_int=is_int)
        self.te_y = DataSet.get_shared(self.np_te_y, is_int=is_int)

        self.tr_x = DataSet.get_shared(self.np_tr_x)
        self.va_x = DataSet.get_shared(self.np_va_x)
        self.te_x = DataSet.get_shared(self.np_te_x)

    def get_datasets():
        return (self.tr_x, self.tr_y, self.v_x, self.v_y, self.te_x, self.te_y)

    @staticmethod
    def get_shared(data, borrow=True, is_int=False):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared = theano.shared(numpy.asarray(data,
                                               dtype=theano.config.floatX),
                                               borrow=borrow)

        if is_int:
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return T.cast(shared, 'int32')
        else:
            return shared


class CSVDataset(DataSet):
    """
    This class reads a CSV file and create a dataset based on that.
    It assumes each row of the file is a (target, feature) set of values.

    Code seperate dataset into three parts (train, validation and test.
    Each part is stored in as (feature, target) tupple.
    Features and targets are numpy 2D arrays, each row represent a sample.

    The output will be saved using Pickle in a tupple format.

    Here we assume that data are randomly ordered. 
    """

    def __init__(self, dataset_file, train_ratio=0.3, valid_ratio=0.2, target_size=1, is_int=False):
        if train_ratio + valid_ratio > 1:
            logging.fatal("Sum of train ratio and valid ratio should be less than or equal to 1")
            raise Exception("Sum of train ratio and valid ratio should be less than or equal to 1")

        if train_ratio < 0 or valid_ratio <0:
            logging.fatal("Train ratio and valid ratio should be more than 0")
            raise Exception("Train ratio and valid ratio should be more than 0")

        with f1 as open(dataset_file, 'rb'):
            logging.info("Reading dataset from CSV file")
            CSVfile = csv.reader(f1)
            ascii_dataset = (row for row in CSVfile)
            float_dataset = ((float(cell) for cell in row) for row in ascii_dataset)

            logging.info("Creating dataset from file's data")
            Targets = (row[0:target_size] for row in float_dataset)
            Features = (row[target_size:] for row in float_dataset)

            numberOfTrain = int(np.floor(len(Targets) * train_ratio))
            numberOfValidation = int(np.floor(len(Targets) * valid_ratio))

            self.np_tr_y = numpy.array(Targets[0:numberOfTrain])
            self.np_va_y = numpy.array(Targets[numberOfTrain:(numberOfTrain+numberOfValidation)])
            self.np_te_y = numpy.array(Targets[(numberOfTrain+numberOfValidation):])

            self.np_tr_x = numpy.array(Features[0:numberOfTrain])
            self.np_va_x = numpy.array(Features[numberOfTrain:(numberOfTrain+numberOfValidation)])
            self.np_te_x = numpy.array(Features[(numberOfTrain+numberOfValidation):])

            AllDataSet = ((self.np_tr_x, self.np_tr_y),
                          (self.np_va_x, self.np_va_y),
                          (self.np_te_x, self.np_te_y),
                          is_int)

            with f2 as open("%s.pkl"%dataset_file, 'wb'):
                logging.info("Writing dataset into a file")
                cPickle.dump(AllDataSet,f2,2)

            logging.info("Converting dataset to shared variable")
            self.make_it_shared(is_int)
            logging.info("Loading dataset is done.")

class NumpyDataset(DataSet):
    """
    This class is for partitioned dataset in following format.
    All self. variables are numpy.2D-array.
    is_int is Boolean.
    AllDataSet = ((self.np_tr_x, self.np_tr_y),
                  (self.np_va_x, self.np_va_y),
                  (self.np_te_x, self.np_te_y),
                  is_int)
    """

    def __init__(self, dataset_file):
        with f as open(dataset_file, 'rb'):
            logging.info("Reading dataset from Pickle file")

            AllDataSet = cPickle.load(f)

            self.np_tr_x, self.np_tr_y = AllDataSet[0]
            self.np_va_x, self.np_va_y = AllDataSet[1]
            self.np_te_x, self.np_te_y = AllDataSet[2]
            is_int = AllDataSet[3]

            logging.info("Converting dataset to shared variable")
            self.make_it_shared(is_int)
            logging.info("Loading dataset is done.")
