import os, math
import utils.load_data as ld

class MarkovChain(object):
    def __init__(self, input_data, inside_model, outside_model):
        '''
        @function:
        to initialize the Markov Chain model, this function will load the input data,
        the inside probability table, and the outside one
        @params:
        input_data, the path of the test input samples
        inside_model: the path of the inside probability table
        outside_model: the path of the outside probability table
        '''
        self.inside_transition_list = ld.load_transition_table(inside_model)
        self.outside_transition_list = ld.load_transition_table(outside_model)

        self.inside_transition_prob_dict = ld.gen_prob_dict(self.inside_transition_list)
        self.outside_transition_prob_dict = ld.gen_prob_dict(self.outside_transition_list)

        self.input_list = ld.load_input_file(input_data)

    def get_prob_list(self):
        '''
        @func:
        to generate the result of the inside_score minus the outside_score
        '''
        inside_result = []
        inside_temp_result = []
        outside_result = []
        outside_temp_result = []
        for sub_list in self.input_list:
            for string in sub_list:
                inside_value = math.log(self.inside_transition_prob_dict.get(string),2)
                inside_temp_result.append(inside_value)
                outside_value = math.log(self.outside_transition_prob_dict.get(string),2)
                outside_temp_result.append(outside_value)
            inside_result.append(inside_temp_result)
            inside_temp_result = []
            outside_result.append(outside_temp_result)
            outside_temp_result = []
        inside = []
        outside = []
        for sub_list in inside_result:
            sum_list = sum(sub_list)
            inside.append(sum_list)
        for sub_list in outside_result:
            sum_list = sum(sub_list)
            outside.append(sum_list)
        self.result = [inside[i]- outside[i] for i in range(0, len(inside))]
        return self.result
    def get_inside_or_outside(self):
        """
        @func:
        after the calculation of the prob_list, generate the final result whether inside of outside
        """
        self.classification_result = []
        for sub in self.result:
            if sub>0:
                value = "Inside"
            if sub<0:
                value = "Outside"
            self.classification_result.append(value)
        return self.classification_result
    def generate_final_result(self):
        """
        @func:
            zip all the final result into a dict
        """
        result_dict = dict(zip(self.result, self.classification_result))
        return result_dict

