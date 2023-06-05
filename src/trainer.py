import sys
import fire
import torch

from utils import set_random_seed
from model.train import MyTrainer
from utils import log_param
from loguru import logger
import warnings
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(action='ignore')

def run_mymodel(device, data_path, param, hyper_param):
    trainer = MyTrainer(device=device,
                        data_path=data_path,
                        )
    trainer.train_with_hyper_param(param=param,
                                   hyper_param=hyper_param)

    logger.info("train has completed")


def main(model='MyArch',
         data_name='MELD',
         seed=0,
         fps=24,
         give_weight=False,
         modal_fusion=False,
         forced_align=False,
         landmark_append=False,
         trans_encoder=False,
         multi_task=False,
         epochs=200,
         act='relu',
         batch_size=1,
         learning_rate=5e-5,
         max_length=60,
         audio_pad_size=60,
         alpha=2,
         dropout=0.2,
         decay_rate=0.98,
         save_at_every=20,
         debug=False,
         ):

    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(seed=seed, device=device)

    param = dict()
    param['model'] = model
    param['device'] = device
    param['data_name'] = data_name
    param['seed'] = seed
    param['fps'] = fps
    param['give_weight'] = give_weight
    param['modal_fusion'] = modal_fusion
    param['forced_align'] = forced_align
    param['landmark_append'] = landmark_append
    param['trans_encoder'] = trans_encoder
    param['multi_task'] = multi_task
    param['save_at_every'] = save_at_every
    param['debug'] = debug
    log_param(param)

    # Step 1. Load datasets
    if data_name == 'MELD':
        data_path = '/home2/dataset/MELD/'
    
    logger.info("path of data is:{}".format(data_path))

    # Step 2. Run (train and evaluate) the specified model
    logger.info("Training the model has begun with the following hyperparameters:")

    hyper_param = dict()
    hyper_param['epochs'] = epochs
    hyper_param['act'] = act
    hyper_param['batch_size'] = batch_size
    hyper_param['learning_rate'] = learning_rate
    hyper_param['max_length'] = max_length
    hyper_param['audio_pad_size'] = audio_pad_size
    hyper_param['alpha'] = alpha
    hyper_param['dropout'] = dropout
    hyper_param['decay_rate'] = decay_rate
    log_param(hyper_param)

    run_mymodel(device=device,
                data_path=data_path,
                param=param,
                hyper_param=hyper_param)
        
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    sys.exit(fire.Fire(main))