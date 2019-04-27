import numpy as np

class EM:
    def __init__(self,data):

        self.data = data

        indice = int(1/2*(len(self.data)))

        self.mean_array = np.zeros(2).reshape(-1,1)
        self.var_array = np.zeros(2).reshape(-1,1)
        self.weight_array = np.zeros(2).reshape(-1,1)
        for i in range(2):
            self.weight_array[i] = 1/2

        self.mean_array[0] = np.mean(self.data[:indice])
        self.mean_array[1] = np.mean(self.data[indice:])
        self.var_array[0] = np.var(self.data[:indice])
        self.var_array[1] = np.var(self.data[indice:])


    def compute_prob(self):
        constant_part = 1/(np.sqrt(self.var_array*2*np.pi))
        exp_part = (np.exp(-((self.data - self.mean_array)**2)/(2*self.var_array)))
        self.prob_matrix = exp_part * constant_part

        E_step = (self.weight_array*self.prob_matrix)/np.sum(self.weight_array*self.prob_matrix, axis = 0)

        M_step_mean = np.sum(E_step*self.data, axis = 1)/np.sum(E_step, axis = 1)
        M_step_var = np.sum(E_step*(self.data - self.mean_array)**2, axis = 1)/np.sum(E_step, axis =1)
        M_step_weight = (1/len(self.data)) * np.sum(E_step, axis = 1)

        if (abs(self.weight_array - M_step_weight)<0.01).any() and (abs(self.mean_array - M_step_mean)<0.01).any() and (abs(self.var_array - M_step_var)<0.01).any() :
            return('over')
        self.mean_array = M_step_mean.reshape(-1,1).copy()
        self.var_array = M_step_var.reshape(-1,1).copy()
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
        class_dict = dict(zip(range(len(self.data)), prob_index_list))
        for i in range(0, 2):
            class_key_list.append([key for key, value in class_dict.items() if value ==i])
            data_group_list.append((self.data[class_key_list[i]]).tolist())
            if len(data_group_list[i]) ==0:
                continue
        return data_group_list
