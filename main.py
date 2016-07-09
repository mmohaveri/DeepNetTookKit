import os
import time

from networks.multi_layer_feed_forward import ShallowNetwork, ConvNetwork
from tools.dataset import CSVDataset

nof_input = 10
nof_middle = 10
l1_reg = 0.00001
l2_reg = 0.0001

t0 = time.time()

# dataset_address = "assets/datasets/train-minst.csv"
#
# if not (os.path.exists(dataset_address+".pkl") and os.path.isfile(dataset_address+".pkl")):
#     # create the numpy dataset
#     my_CSV_dataset = CSVDataset(dataset_address, is_int=True, normalization_factor=255)
#
# my_network = ShallowNetwork(input_size=784, nof_middle=20, nof_output=10, l1_reg=0.00001, l2_reg=0.0001)
#
# my_network.train(dataset_address+".pkl")
#
# my_network.save_network("assets/networks/test.pkl")
#





dataset_address = "assets/datasets/train-minst.csv"

if not (os.path.exists(dataset_address+".pkl") and os.path.isfile(dataset_address+".pkl")):
    # create the numpy dataset
    my_CSV_dataset = CSVDataset(dataset_address, is_int=True, normalization_factor=255)

my_network = ConvNetwork(input_shape=(28, 28), nof_output=10, l1_reg=l1_reg, l2_reg=l2_reg)

my_network.train(dataset_address+".pkl")

my_network.save_network("assets/networks/test.pkl")

t1 = time.time()

print "Computation ended in %d seconds" % (t1-t0)
