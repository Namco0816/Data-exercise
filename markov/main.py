import math, shutil, os, argparse
from markovChain import MarkovChain
from HMM import Viterbi
def str2bool(v):
    if v.lower() in ("y", "yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "n", "f", "false", "0"):
        return False
    else:
        return argparse.ArgumentTypeError("Bool typl input expected")

parser = argparse.ArgumentParser(description = "Markov Chain and Markov Hidden Model")
parser.add_argument("--input_data_path", help = "path to processed data")
parser.add_argument("--inside_model_path",help = 'path of the inside model' )
parser.add_argument('--outside_model_path', help = 'path of the outside model')
parser.add_argument("--markov_chain", type = str2bool, nargs = '?', const = True, default = False, help = "start the markov chain model")
parser.add_argument("--markov_hidden", type = str2bool, nargs = "?", const = True, default = False, help = "start the hidden markov model")
args = parser.parse_args()

doMarkovChain = args.markov_chain
doHMM = args.markov_hidden

def main():
    global args
    if doMarkovChain:
        model = MarkovChain(args.input_data_path, args.inside_model_path, args.outside_model_path)
        prob_list = model.get_prob_list()
        classification_result = model.get_inside_or_outside()
        final_result = model.generate_final_result()
        print(final_result)
    if doHMM:
        model = Viterbi(args.input_data_path)
        model.load_input_data()#TODO
if __name__ == "__main__":
    main()
    print("DONE")
