from parse import parse_args
from utils import fix_seeds
import os
import torch

if __name__ == '__main__':
    args, special_args = parse_args()
    print(args)

    fix_seeds(args.seed) # set random seed

    rs_type = args.rs_type # LLM, Seq, General, etc.
    print(f'from models.{rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS')
    # from models.General.UniSRec import UniSRec_RS
    try:
        exec(f'from models.{args.rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS') # load the model
    except:
        print('Model %s not implemented!' % (args.model_name))
    
    # RS = UniSRec_RS(args, special_args)
    # from models.General.IntentCF import IntentCF_RS
    # RS = IntentCF_RS(args, special_args)
    try:
        RS = eval(args.model_name + '_RS(args, special_args)') # load the recommender system
    except:
        RS = eval(args.model_name + '_RS(args)')

    RS.execute() # train and test
    print('Done!')

