import utils.load_HMM_data as ld
import math
class Viterbi(object):
    def __init__(self, prior_data_path):
        '''
        @func:
            to load the prior probability data,and the symbol.
        '''
        self.path = []
        self.s1_node_recorder = {}
        self.s2_node_recorder = {}
        self.index = 0
        self.decripition, self.initial_prob, self.transition_state, self.symbol_prob = ld.load_hmm_data(prior_data_path)
    def load_input_data(self, input_data_path):
        '''
        @func:
            load the test samples
        '''
        with open(input_data_path) as f:
            data = [line.strip() for line in f]
        self.new_data = "".join(data[1:])
        self.new_data = list(self.new_data)
    def update(self):
        """
        @func:
            after load the data, try to update the model state step by step
        """
        self.current_item = self.new_data[self.index]
        s1_symbol_key = "s1_"+self.current_item.lower()
        s2_symbol_key = "s2_"+self.current_item.lower()
        self.s1_symbol_prob = self.symbol_prob.get(s1_symbol_key)
        self.s2_symbol_prob = self.symbol_prob.get(s2_symbol_key)
        '''
        start calculate the score
        '''
        if self.index ==0:
            self.current_s1_score = math.log(self.initial_prob.get('state_1_init_prob') * self.s1_symbol_prob,2)
            self.s1_node_recorder['step_%d'%(self.index)]={'score': self.current_s1_score, "previous": None}
            self.current_s2_score = math.log(self.initial_prob.get('state_2_init_prob') * self.s2_symbol_prob,2)
            self.s2_node_recorder['step_%d'%(self.index)]={'score': self.current_s2_score, "previous": None}
        else:
            if self.previous_s1_score+math.log(self.transition_state.get('s1_2_s1')*self.s1_symbol_prob,2)>self.previous_s2_score+math.log(self.transition_state.get('s2_2_s1')*self.s1_symbol_prob,2):
                self.current_s1_score = self.previous_s1_score+math.log(self.transition_state.get('s1_2_s1')*self.s1_symbol_prob,2)
                self.s1_node_recorder['step_%d'%(self.index)]={'score': self.current_s1_score, "previous": "s1"}
            else:
                self.current_s1_score = self.previous_s2_score+math.log(self.transition_state.get('s2_2_s1')*self.s1_symbol_prob,2)
                self.s1_node_recorder['step_%d'%(self.index)]={'score': self.current_s1_score, "previous": "s2"}

            if self.previous_s2_score+math.log(self.transition_state.get('s2_2_s2')*self.s2_symbol_prob,2)>self.previous_s1_score+math.log(self.transition_state.get('s1_2_s2')*self.s2_symbol_prob,2):
                self.current_s2_score = self.previous_s2_score+math.log(self.transition_state.get('s2_2_s2')*self.s2_symbol_prob,2)
                self.s2_node_recorder['step_%d'%(self.index)]={'score': self.current_s2_score, "previous": "s2"}
            else:
                self.current_s2_score = self.previous_s1_score+math.log(self.transition_state.get('s1_2_s2')*self.s2_symbol_prob,2)
                self.s2_node_recorder['step_%d'%(self.index)]={'score': self.current_s2_score, "previous": "s1"}

        self.index+=1
        self.previous_s1_score = self.current_s1_score
        self.previous_s2_score = self.current_s2_score

    def get_len(self):
        '''
        @func:
            return the length of the whole test samples.
        '''
        return len(self.new_data)

    def gen_result(self):
        self.index = self.index - 1
        flag = None
        if self.s1_node_recorder['step_%d'%self.index].get('score')>self.s2_node_recorder['step_%d'%self.index].get('score'):
            flag = 's1'
        else:
            flag = 's2'
        for i in range(self.index, 0,-1):
            if i == 1:
                print("the data index which bigger than 0 are with the {} state".format(flag))
            if flag =='s1':
                path = self.s1_node_recorder['step_%d'%i].get("previous")
                if path =='s1':
                    continue
                if path =='s2':
                    print("the data index which bigger than {} are with the {} state".format(i, flag))
                    flag = 's2'
                    continue
            if flag =='s2':
                path = self.s2_node_recorder["step_%d"%i].get('previous')
                if path =='s2':
                    continue
                if path =='s1':
                    print("the data index which bigger than {} are with the {} state".format(i, flag))
                    flag ='s1'
                    continue

