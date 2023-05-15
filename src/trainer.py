import os.path
import sys
import fire
import torch
from pathlib import Path
import pandas as pd

from utils import set_random_seed
from model.train import MyTrainer
from utils import log_param
from loguru import logger


def run_mymodel(device, data_path, param, hyper_param):
    trainer = MyTrainer(device=device,
                        data_path=data_path,
                        )
    model = trainer.train_with_hyper_param(param=param,
                                           hyper_param=hyper_param)

    logger.info("train has completed")
    return 100.0
    evaluator = MyEvaluator(device=device)
    loss = evaluator.evaluate(model, 
                              data_path=data_path
                              )
    return loss


def main(model='model',
         data_name='MELD',
         seed=0,
         fps=24,
         give_weight=True,
         modal_fusion=True,
        #  max_history=10,
         epochs=100,
         act='relu',
         batch_size=1,
         learning_rate=1e-3,
         max_length=60,
         alpha=2,
         dropout=0.2,
         decay_rate=0.98,
         save_at_every=10,
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
    param['save_at_every'] = save_at_every
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
    hyper_param['alpha'] = alpha
    hyper_param['dropout'] = dropout
    hyper_param['decay_rate'] = decay_rate
    # hyper_param['max_history'] = max_history
    log_param(hyper_param)

    if model == 'model':
        loss = run_mymodel(device=device,
                            data_path=data_path,
                            param=param,
                            hyper_param=hyper_param)

        # - If you want to add other model, then add an 'elif' statement with a new runnable function
        #   such as 'run_my_model' to the below
        # - If models' hyperparamters are varied, need to implement a function loading a configuration file
    else:
        logger.error("The given \"{}\" is not supported...".format(model))
        return

    # Step 3. Report and save the final results
    logger.info("The model has been trained. The test loss is {:.4} .".format(loss))


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    sys.exit(fire.Fire(main))