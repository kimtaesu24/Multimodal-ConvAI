# Multimodal-ConvAI

## Repository Structure

The overall file structure of this repository is as follows:

```
Multimodal-ConvAI
    ├── README.md                       
    ├── requirments
    ├── run.py                          # starts training the model with specified hyperparameters
    └── src         
        ├── utils.py                    # contains utility functions such as setting random seed and showing hyperparameters
        ├── trainer.py                  # processes input arguments of a user for training
        ├── inference.py                # implements a function for inference the model
        ├── baseline.py                 # original dialogpt running code
        └── models                      
            ├── architecture.py        # implements the forward function of the architecture
            ├── arch_data.py           # loda dataset for dataloader
            ├── train.py               # implements a function for training the model with hyperparameters
            ├── DAN
            └── Forced_Alignment
```

## How To Run

You can simply check if the model works correctly with the following command:
```
PYTHONPATH=src python3 run.py --arch_name $ARCHITECTURE
```
The above command will start learning the model on the `$ARCHITECTURE` with the specified parameters saved in `param.json`.