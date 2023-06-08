import os
import fire
import sys
import json
from dotmap import DotMap
from src.trainer import main

os.chdir('./src')


def main_wrapper(arch_name=''):
    param_path = f'../hyperparameter/{arch_name}param.json'

    with open(param_path, 'r') as in_file:
        param = DotMap(json.load(in_file))

    main(model=param.model,
         data_name=param.data_name,
         seed=param.seed,
         fps=param.fps,
         give_weight=param.give_weight,
         modal_fusion=param.modal_fusion,
         forced_align=param.forced_align,
         landmark_append=param.landmark_append,
         trans_encoder=param.trans_encoder,
         multi_task=param.multi_task,
         epochs=param.epochs,
         act=param.act,
         batch_size=param.batch_size,
         learning_rate=param.learning_rate,
         max_length=param.max_length,
         history_length=param.history_length,
         audio_pad_size=param.audio_pad_size,
         alpha=param.alpha,
         dropout=param.dropout,
         decay_rate=param.decay_rate,
         save_at_every=param.save_at_every,
         debug=param.debug,
         )


if __name__ == "__main__":
    sys.exit(fire.Fire(main_wrapper))
