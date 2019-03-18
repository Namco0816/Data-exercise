def load_transition_table(data_path):
    transition_list = []
    with open (data_path) as f:
        data = f.readlines()
    for line in data:
        prob_list = line.split()
        prob_list = list(map(float,prob_list))
        transition_list += prob_list
    return transition_list

def load_input_file(input_data_path):
    str_list = []
    split_num = 2
    with open(input_data_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip('\n')
        new_line = [line[i:i+split_num] for i in range(0, len(line))]
        str_list.append(new_line)
    for i in str_list:
        for j in i:
            if len(j) < 2:
                i.remove(j)
    return str_list

def gen_prob_dict(transition_list):
    keys = ['AA', 'AC', 'AG', 'AT',
            'CA', 'CC', 'CG', 'CT',
            'GA', 'GC', 'GG', 'GT',
            'TA', 'TC', 'TG', 'TT']

    transition_prob_dict = dict(zip(keys,transition_list))
    return transition_prob_dict

