from EM import EM
from Kmeans import Kmeans

import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'Kmeans and EM algorithm for 1 dimension data')
parser.add_argument('--method', type = str)
parser.add_argument('--mean_1', type = int, default = 10, help = 'The first groups mean')
parser.add_argument('--mean_2', type = int, default = 100, help = 'The second groups mean')
parser.add_argument('--std_1', type = int, default = 1, help = 'The first groups std')
parser.add_argument('--std_2', type = int, default = 10, help = 'The second groups std')
parser.add_argument('--count_1', type = int, default = 100, help = 'The first groups data amount')
parser.add_argument('--count_2', type = int, default = 100, help = 'The second groups data amount')
parser.add_argument('--k' , type = int, default = 2, help = 'start with k groups of cluster')
parser.add_argument('--iter_rounds', type = int, default = 100, help = "The total iteration rounds")
parser.add_argument('--seed', type = int, default = 1234, help ="the random seed")
args = parser.parse_args()

def main():
    global args
    np.random.seed(args.seed)
    data_1 = np.random.normal(args.mean_1, args.std_1, args.count_1)
    data_2 = np.random.normal(args.mean_2, args.std_2, args.count_2)
    data = np.hstack((data_1, data_2))
    if args.method.lower() == 'kmeans':
        cluster = Kmeans(k = args.k, data = data)
    if args.method.lower() == 'em':
        cluster = EM(k = args.k, data = data)
    result = cluster.start_iter(args.iter_rounds)
    for i in range(0, args.k):
        print('GROUPS NO.{}: '.format(i+1), result[i], '\t')
if __name__ =='__main__':
    main()
    print ('DONE')
