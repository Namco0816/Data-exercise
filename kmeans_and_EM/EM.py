import numpy as np

class EM:
    def __init__(self, k, data):

        self.sorted_data = np.sort(data)
        self.k = k
        indices = int(len(self.sorted_data)/self.k)

        self.data_group_list = []

        self.mean_array = np.zeros(k).reshape(-1,1)
        self.std_array = np.zeros(k).reshape(-1,1)

        for i in range(0, k):
            self.mean_array[i] = np.mean(self.sorted_data[i*indices:(i+1)*indices])
            self.std_array[i] = np.std(self.sorted_data[i*indices:(i+1)*indices])

    def compute_prob(self):
        class_key_list = []
        self.prev_data_group = self.data_group_list.copy()
        self.data_group_list = []
        constant_part = 1/(self.std_array*np.sqrt(2*np.pi))
        exp_part = (np.exp(-((self.sorted_data - self.mean_array)**2)/(2*self.std_array**2)))
        prob_index_list = np.argmax(constant_part * exp_part, axis = 0)

        self.class_dict = dict(zip(range(len(self.sorted_data)), prob_index_list))
        for i in range(0, self.k):
            class_key_list.append([key for key, value in self.class_dict.items() if value ==i])
            self.data_group_list.append((self.sorted_data[class_key_list[i]]).tolist())
            if len(self.data_group_list[i]) ==0:
                continue
            self.mean_array[i] = np.mean(self.data_group_list[i])
            self.std_array[i] = np.std(self.data_group_list[i])
        if (self.prev_data_group == self.data_group_list):
            return('over')
    def start_iter(self, iter_count):
        for i in range(iter_count):
            iter_flag = self.compute_prob()
            if iter_flag == 'over':
                print("AFTER {}'S ROUNDS OF PROCESS, THE ALGORITHM CONVERGES, STOP CLUSTERING".format(i))
                break
        return self.data_group_list
