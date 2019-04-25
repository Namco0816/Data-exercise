import numpy as np

class Kmeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data

        self.k_point_array = np.zeros(k).reshape(-1, 1)
        self.prev_k_point_array = np.zeros(k).reshape(-1, 1)

        self.group_list = []

        max_index = np.argmax(self.data)
        self.max = self.data[max_index]

        min_index = np.argmin(self.data)
        self.min = self.data[min_index]

        indice = (self.max - self.min)/self.k
        for i in range(0, k):
            self.k_point_array[i] = indice*i

    def compute_centroid(self):
        self.prev_k_point_array = self.k_point_array.copy()
        group_key_list = []
        self.group_list = []
        distance_matrix = np.abs(self.data - self.k_point_array)
        class_matrix = np.argmin(distance_matrix, axis =0)

        index_list = range(len(self.data))
        class_list = class_matrix.tolist()

        self.class_dict = dict(zip(index_list, class_list))
        for i in range(0, self.k):
           group_key_list.append([key for key,value in self.class_dict.items() if value ==i])
           self.group_list.append(self.data[group_key_list[i]])
           if len(self.group_list[i]) == 0:
               continue
           self.k_point_array[i] = sum(self.group_list[i])/len(self.group_list[i])
        if ((self.prev_k_point_array == self.k_point_array).all()):
            return ('over')
    def start_iter(self, iter_count):
        for i in range(iter_count):
            iter_flag = self.compute_centroid()
            if iter_flag == 'over':
                print("AFTER {}'S ROUNDS OF PROCESS, THE ALGORITHM CONVERGES, STOP CLUSTERING".format(i))
                break
        return self.group_list
