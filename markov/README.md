#Markov Chain and Hidden Markov Model

A simple implementation and application of ***Markov Chain*** and ***Viterbi dp algorithm***
---
**markovChain.py**: the Markov Chain file
**HMM.py** : the viterbi model file
***QUICK START of MARKOV CHAIN***:
`python main.py --input_data_path [MARKOV INPUT TEST INSTANCE] --inside_model_path [CPG ISLAND INSIDE MODEL FILE] --outside_model_path [CPG ISLAND OUTSIDE MODEL FILE] --markov_chain True`
for example you can use:
`python main.py --input_data_path ./data/test1.txt --inside_model_path ./data/inside.txt --outside_model_path ./data/outside.txt --markov_chain True`
***QUICK START OF HMM VITERBI:***
`python main.py --input_data_path [HMM INPUT PATH] --hmm_info_path [HMM INFO PATH] --markov_hidden True`
for example you can:
`python main.py --input_data_path ./data/example.fa --hmm_info_path ./data/example.hmm --markov_hidden True`

see more info by using:
`python main.py --help`

Author:@TanHaochen
