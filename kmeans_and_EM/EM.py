import numpy as np

class EM:
    def __init__(self, k, data):

        self.sorted_data = np.sort(data)
        self.k = k
        indices = int(len(self.sorted_data)/self.k)


        self.mean_array = np.zeros(k).reshape(-1,1)
        self.std_array = np.zeros(k).reshape(-1,1)
        self.weight_array = np.zeros(k).reshape(-1,1)
        for i in range(self.k):
            self.weight_array[i] = 1/self.k

        for i in range(0, k):
            self.mean_array[i] = np.mean(self.sorted_data[i*indices:(i+1)*indices])
            self.std_array[i] = np.std(self.sorted_data[i*indices:(i+1)*indices])


    def compute_prob(self):
        constant_part = 1/(self.std_array*np.sqrt(2*np.pi))
        exp_part = (np.exp(-((self.sorted_data - self.mean_array)**2)/(2*self.std_array**2)))
        self.prob_matrix = exp_part * constant_part

        E_step = self.weight_array*self.prob_matrix/np.sum(self.weight_array*self.prob_matrix, axis = 0)
        for items in E_step:
            if (items == 0).all():
                return('bad_k')
        M_step_mean = np.sum(E_step*self.sorted_data, axis = 1)/np.sum(E_step, axis = 1)
        M_step_std = np.sum(E_step*(self.sorted_data - self.mean_array)**2, axis = 1)/np.sum(E_step, axis =1)
        M_step_weight = (1/len(self.sorted_data)) * np.sum(E_step, axis = 1)

        if (abs(self.weight_array - M_step_weight)<0.001).any() and (abs(self.mean_array == M_step_mean)<0.001).any():
            return('over')
        self.mean_array = M_step_mean.reshape(-1,1).copy()
        self.std_array = M_step_std.reshape(-1,1).copy()
        self.weight_array = M_step_weight.reshape(-1,1).copy()

    def start_iter(self, iter_count):
        class_key_list = []
        data_group_list = []
        for i in range(iter_count):
            iter_flag = self.compute_prob()
            if iter_flag == 'over':
                print("AFTER {}'S ROUNDS OF PROCESS, THE ALGORITHM CONVERGES, STOP CLUSTERING\t" .format(i))
                break
            if iter_flag =='bad_k':
                print("AFTER {}'S ROUNDS OF ITERATION THE PROCESS STOPPED DUE TO THE BAD INITIALIZE OF K, TRY DECREASE THE VALUE OF K TO GET BETTER RESULT\t")
                break
        prob_index_list = np.argmax(self.prob_matrix, axis = 0)
        class_dict = dict(zip(range(len(self.sorted_data)), prob_index_list))
        for i in range(0, self.k):
            class_key_list.append([key for key, value in class_dict.items() if value ==i])
            data_group_list.append((self.sorted_data[class_key_list[i]]).tolist())
            if len(data_group_list[i]) ==0:
                continue
        return data_group_list
