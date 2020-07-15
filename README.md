# MADPQN - A Manifold-Aware Distributed Proximal Quasi-Newton Method for Regularized Optimization

This code implements the algorithm proposed in the following paper in C/C++ and MPI:
_LI Yu-Sheng, CHIANG Wei-Lin, LEE Ching-pei. [Manifold Identification for Ultimately Communication-Efficient Distributed Optimization] (http://www.optimization-online.org/DB_FILE/2020/06/7833.pdf). The 37th International Conference on Machine Learning, 2020._

## Getting started
To compile the code, you will need to install g++, and an implementation of MPI.
You will need to list the machines being used in a separate file, and make sure they are directly accessible through ssh.
Additionally the code depends on the BLAS and LAPACK libraries.

The code split.py, borrowed from [MPI-LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/), partition the data and distribted the segments to the designated machines.
Then the program ./train solves the optimization problem to obtain a model.

## Problem being solved

The code solves
(1): the L1-regularized logistic regression problem.

min_{w} |w|_1 + C \sum_{i=1}^n \log(1 + \exp(- y_i w^T x_i))

with a user-specified parameter C > 0,

(2): the L1-Regularized least-sqaure regression (LASSO) problem.

min_{w} |w|_1 + C \sum_{i=1}^n (w^T x_i - y_i)^2 / 2.

with a user-specified parameter C > 0,

(3): the L1-regularized L2-loss support vector classification problem.

min_{w} |w|_1 + C \sum_{i=1}^n \max\{0,1 - y_i w^T x_i, 0\}^2.

with a user-specified parameter C > 0,

and
(4): the GroupLASSO-regularized multinomial logistic regression problem for multi-class classification.
