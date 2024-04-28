# FBLG: A Local Graph Based Approach for Handling Dual Skewed Non-IID Data in Federated Learning

This is the PyTorch implementation of our [IJCAI 2024](https://ijcai24.org/) paper: FBLG: A Local Graph Based Approach for Handling Dual Skewed Non-IID Data in Federated Learning

## ChangeLog
* [2024-04-28] Code released with reimplementations of experiments

## Preparation

### Environment
* python/3.10.9
* pytorch/1.12.1
* cuda/11.3.1
* numpy/1.23.5
* scipy/1.10.0

### Dataset
* Due to size restrictions, RAW_DATA is at https://drive.google.com/drive/folders/1-pcuOjUsFI64kHoAJJ6_D9I1PPfQtmhw?usp=drive_link
* MNIST
* FASHION
* CIFAR10
* SVHN

## Training
### generate Non-IID data

* generate the case where the label distribution and sample size are both skewed among clients
```python 
generate_fedtask.py  --benchmark mnist_classification --dist 4 --skew 0.79 --num_clients 20
```

* generate the case where only the label distribution is skewed among clients
```python 
generate_fedtask.py  --benchmark mnist_classification --dist 3 --skew 0.4 --num_clients 20
```

* generate the case where only the sample size is skewed among clients
```python 
generate_fedtask.py  --benchmark mnist_classification --dist 4 --skew 0.6 --num_clients 20
```

### run FBLG algorithm
```python 
main.py  --task mnist_classification_cnum20_dist4_skew0.79_seed0 --algorithm FBLG --num_rounds 300 --gpu 0
```

## Supplementary experiments

### verify the impact of different numbers of clients on algorithm performance during rebuttal
|      Algorithm      |Round | Number of clients | MNIST | FMNIST | SVHN  | Number of clients  | MNIST | FMNIST  |  SVHN  |
|:-------------------:|:----:|:-----------------:|:-----:|:------:|:-----:|:------------------:|-------|:-------:|:------:|
|       FedAvg        | 300  |        50         | 97.84 | 81.56  | 66.07 |        100         | 97.59 |  80.21  | 78.23  |
|      MDSample       | 300  |        50         | 97.54 | 78.81  | 65.57 |        100         | 97.56 |  81.81  | 79.04  |
|   Power-Of-Choice   | 300  |        50         | 98.06 | 59.32  | 58.93 |        100         | 98.11 |  70.88  | 61.67  |
|  FedProx (μ = 0.1)  | 300  |        50         | 96.70 | 79.13  | 59.77 |        100         | 97.19 |  80.80  |  78.26 |
|        Moon         | 300  |        50         | 95.91 | 79.02  | 51.25 |        100         | 96.29 |  77.22  | 64.78  |
|      Scaffold       | 300  |        50         | 93.26 | 76.76  | 55.32 |        100         | 95.62 |  75.07  | 60.84  |
|       FedAvgM       | 300  |        50         | 97.71 | 78.95  | 68.97 |        100         | 97.63 |  79.92  | 72.92  |
|       FedNova       | 300  |        50         | 97.78 | 78.89  | 61.91 |        100         | 96.95 |  81.08  | 57.61  |
|        FedGS        | 300  |        50         | 97.44 | 80.12  | 54.09 |        100         | 97.25 |  78.33  | 73.06  |
|   FBLG (C = 0.5)    | 300  |        50         | 97.79 | 82.88  | 75.11 |        100         | 98.30 |  80.67  | 83.35  |
|   FBLG (C = 0.4)    | 300  |        50         | 97.92 | 83.25  | 63.29 |        100         | 98.10 |  81.32  | 84.28  |

## Citation

If you find this work useful, please consider citing it.
