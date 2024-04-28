import benchmark.visualization
from benchmark.svhn_classification.model import cnn
import benchmark.partition

default_partitioner = benchmark.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = cnn
visualize = benchmark.visualization.visualize_by_class