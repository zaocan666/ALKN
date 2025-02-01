# Project Overview

The proposed method mitigates utility decline in LLM continual unlearning scenarios. The method dynamically adjusts the unlearning intensity by using adaptive training objectives with soft labels based on initial prediction probabilities. We apply a dynamic mask on the gradients of model parameters. The mask is learned from the data. We assign specific learning rates to individual parameters, further mitigating utility loss during the training process.

## Files in this Workspace

- `finetune.py`: Script for fine-tuning the model.
- `forget_CL.py`: Script for forgetting specific tasks.
- `ALKN.py`: Custom trainer for the proposed method.
- `evaluate_util_CL.py`: Utility functions for evaluating the model.
- `config`: Configuration files for setting parameters.
- `task_vector.py`: Module for handling task vectors.
- `utils.py`: Utility functions used throughout the project.

## Reference

This code is based on [TOFU](https://github.com/locuslab/tofu).

