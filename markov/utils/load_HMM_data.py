def load_hmm_data(data_path):
    new_list = []
    with open(data_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line= line.split()
        new_list.append(line)
    description_key = ['state Count', 'symbol Count', 'symbol']
    description_dict = dict(zip(description_key, new_list[0]))

    initial_prob_key = ['state_1_init_prob', 'state_2_init_prob']
    initial_prob_dict = dict(zip(initial_prob_key, [float(new_list[1][i]) for i in range(len(new_list[1]))]))

    transition_list = new_list[2][0:2]+new_list[3][0:2]
    transition_state_key = ['s1_2_s1', 's1_2_s2', 's2_2_s1', 's2_2_s2']
    transition_state_dict = dict(zip(transition_state_key, [float(i) for i in transition_list]))

    prob_symbol_list = new_list[2][2:]+new_list[3][2:]
    prob_symbol_key = ['s1_a','s1_c','s1_g','s1_t','s2_a', 's2_c', 's2_g', 's2_t']
    prob_symbol_dict = dict(zip(prob_symbol_key, [float(i) for i in prob_symbol_list]))
    return description_dict, initial_prob_dict, transition_state_dict, prob_symbol_dict

