# Introduction

- Implementation of K-means and EM algorithm for the 1-dim normal distribution data.
- Run quick test for EM and K-means, the default data_1(std 1, mean 5, count 100), default data_2(std2, mean 10, count 100):

***`python main.py --method kmeans`*** 

and 

***`python main.py --method em`***

- Customize data to do the test:
`python main.py --method [KMEANS or EM] --mean_1 [MEAN OF DATA_1] --std_1 [STD of DATA_1] --mean_2 [MEAN OF DATA_2] --std_2[STD OF DATA_2] --seed [RANDOM SEED] --count_1 [NUM of DATA_1] --count_2 [NUM OF DATA_2]`

# K-MEANS

See k-means code in **Kmeans.py**

```python
__init__(k, data)
'''
:func: initialize function, eg. data = [1,2,3,4,5,6,7,8], k = 2, 
then k_point array = [1* (8 -1)/2, 2* (8-1)/2]
:params k: number of groups
:params data: input data
'''
compute_centroid(self)
'''
:func: compute the centroid of each groups and update the groups member
'''
start_iter(iter_rounds)
'''
:func: do the params updating iteration
:params iter_rounds: define the max iteration rounds of the computation, 
if the iteration rounds bigger than the iter_rounds and still with no converge, 
stop the iteration.
'''
```
    
# EM

See codes in EM.py
   
```python
__init__(data)
'''
:func:  initialization function
:params data: input data
'''
compute_prob()
'''
:func: consist of two parts, with first step E step to guess the distribution, 
second step M step to update the params.
'''
start_iter(iter_rounds)
'''
:func: do the params updating iteration
:params iter_rounds: define the max iteration rounds of the computation, 
if the iteration rounds bigger than the iter_rounds and still with no converge,
stop the iteration.
'''
```
    
The threshold has been set to 0.01, if the difference between olf params(var, mean, weight) smaller than the threshold, return the result and stop iteration.

    
# Comparision
For the k-means algorithm, when testing with dataset1(std 1, mean 5, count 100) and dataset2(std2, mean 10, count 100), the result is as below:

ACC:  0.925
ORGINAL MEAN: [5, 10]
ORIGNAL STD:  [1, 2]

FINAL MEAN:  [5.262785246536643, 10.450118827464818]
FINAL STD:  [1.121827191712395, 1.5548772509323725]

For the EM algorithms, the result is:
ACC:  0.945
ORGINAL MEAN: [5, 10]
ORIGNAL STD:  [1, 2]

FINAL WEIGHT: [[0.53154758 0.46845242]]
FINAL MEAN: [[ 5.14485332 10.10277139 ]]
FINAL STD: [[1.06568105 1.84965942]]

When 2 datasets with same std, this two algorithms performs similary. However for the datasets with difference of the std, the EM outperforms the K-means. see more comparision as below:

![](./same_std.png)
![](./diff_std_diff_mean.png)
![](./same_mean.png)

# Conclusion:
K-means is really a fast algorithm but with lots of disadvantages:
- K-means algorithm is actually a process of convex optimization, for the non-convex problem, performance not good
- It takes K-means for a long time to converges to a local-optimal solution, which for the EM algorithm is quiet easy. Thus for a complex datasets, the EM converges much more faster than k-means.

@author Tan Haochen
