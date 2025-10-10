import argparse
from datetime import datetime, timedelta, timezone
import yaml
import warnings
import logging

warnings.filterwarnings("ignore")

import numpy
import random
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from exp.exp_GraphTransformer import ExpGraphTransformer

'''
nohup python ./main.py --config=model_config_xian_7_20inf.yaml --dis_type=dtw --seed=2000 &> result &
'''

def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajGAT")
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-G", "--gpu", type=str, default="0")
    parser.add_argument("-L", "--load-model", type=str, default=None)
    parser.add_argument("-J", "--just_embedding", action="store_true")
    parser.add_argument("--dis_type", type=str)
    parser.add_argument("--seed", type=int, default=2000)

    args = parser.parse_args()

    dt = datetime.now(timezone(timedelta(hours=10)))
    logfile = dt.strftime("%Y%m%d_%H%M%S") + '.log'
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler('./log/trajgat_'+logfile, mode = 'w'), 
                        logging.StreamHandler()]
            )


    with open(args.config, "r") as f:
        config = yaml.load(f)
        if args.dis_type != "":
            config['dis_type'] = args.dis_type
            
        config['seed'] = args.seed
        set_seed(config['seed'])

        config['traj_path'] = './data/{}_TrajGAT_trajsimi_traj.pkl'.format(config['data'])
        config['dis_matrix_path'] = './data/{}_TrajGAT_trajsimi_{}_simi.pkl'.format(config['data'], config['dis_type'])


    logging.info("Args in experiment:")
    logging.info(config)
    logging.info("Load model:{}".format(args.load_model))
    logging.info("Store embeddings:{}\n".format(args.just_embedding))

    if args.just_embedding:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).embedding()
    else:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).train()

    torch.cuda.empty_cache()
