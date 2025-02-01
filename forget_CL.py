from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA, TextForgetDatasetQA_CL, TextDatasetQA_CL
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
from ALKN import ALKN_trainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, TrainerCallback, TrainerState, TrainerControl
from transformers.modeling_utils import PreTrainedModel

import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from task_vector import TaskVector
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
from evaluate_util_CL import custom_data_collator_with_indices

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget_CL")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    else:
        local_rank = 0
    
    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]


    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    torch_format_dataset = TextForgetDatasetQA_CL(cfg.data_path, task_ids=cfg.task_ids, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, loss_type=cfg.forget_loss)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    print(f"task_ids: {cfg.task_ids}")

    # Load eval_dataloader
    eval_dataset = TextDatasetQA_CL(
        cfg.data_path,
        tokenizer=tokenizer, 
        task_id=cfg.task_ids[-1],
        model_family=cfg.model_family, 
        max_length=cfg.eval.generation.max_length, 
        split="forget10_perturbed", 
        question_key="question", 
        answer_key="answer"
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=cfg.eval.batch_size, collate_fn=custom_data_collator_with_indices
    )

    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break
    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
        if (cfg.forget_loss == "KL") or ("npo" in cfg.forget_loss) or cfg.forget_loss == "task_vector" or cfg.forget_loss == "CTV":
            oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    mask_previous = None
    mask_threshold = None
    if cfg.forget_loss == "CTV" and cfg.task_ids[-1] > 0:
        assert cfg.model_path.endswith("substracted_model/")
        mask_dir = os.path.join(cfg.model_path[:-len("substracted_model/")], 'CTV_mask')
        mask_path = os.path.join(mask_dir, 'CTV_mask.pt')
        if os.path.exists(mask_path):
            mask_previous = torch.load(mask_path)
            mask_threshold = torch.load(os.path.join(mask_dir, 'mask_threshold.pt'))
        else:
            raise FileNotFoundError(f"Mask file not found at {mask_path}")
    
    cfg.lr = cfg.lr * (cfg.lr_decay ** cfg.task_ids[-1])
    cfg.tv_scaling_coef = cfg.tv_scaling_coef * (cfg.coef_decay ** cfg.task_ids[-1])

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            log_level="debug",
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="no", #"steps" if cfg.save_model and (not cfg.eval_only) else "no",
            ddp_find_unused_parameters= False,
            weight_decay = cfg.weight_decay,
            eval_steps = cfg.eval_steps,
            evaluation_strategy = "steps" if cfg.eval_steps>0 else "no",
            seed=cfg.seed,
            eval_on_start=(cfg.eval_steps>0),
        )
    
    if cfg.forget_loss != "CTV":
        trainer = CustomTrainerForgetting(
            model=model,
            tokenizer=tokenizer,
            train_dataset=torch_format_dataset,
            eval_dataset = torch_format_dataset,
            compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
            my_eval_dataloader = eval_dataloader,
            eval_threshold = cfg.forget_rouge_threshold,
            args=training_args,
            data_collator=custom_data_collator_forget,
            oracle_model = oracle_model,
            forget_loss = cfg.forget_loss,
            eval_cfg = cfg.eval,
            beta = cfg.beta,
            npo_coeff=cfg.npo_coeff,
            grad_diff_coeff=cfg.grad_diff_coeff,
            KL_coeff=cfg.KL_coeff,
            performance_threshold=cfg.forget_rouge_threshold,
        )
    else:
        trainer = ALKN_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=torch_format_dataset,
            eval_dataset = torch_format_dataset,
            compute_metrics=None,
            args=training_args,
            data_collator=custom_data_collator_forget,
            oracle_model = oracle_model,
            forget_loss = cfg.forget_loss,
            unmask_ratio = cfg.unmask_ratio,
            mask_threshold = mask_threshold,
            threshold_update_step = cfg.threshold_update_step,
            mask_previous=mask_previous,
            low_lr_ratio=cfg.low_lr_ratio,
            mask_p_k=cfg.mask_p_k,
            init_prob_k=cfg.init_prob_k
        )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    if cfg.save_model and (not cfg.eval_only):
        if cfg.forget_loss != "task_vector" and cfg.forget_loss != "CTV":
            trainer.save_model(cfg.save_dir)
            trainer.state.save_to_json(os.path.join(cfg.save_dir, "trainer_state.json"))
        else:
            print("Saving task vector")
            tv_save_dir = os.path.join(cfg.save_dir, 'task_vector')
            trainer.save_model(tv_save_dir)
            trainer.state.save_to_json(os.path.join(tv_save_dir, "trainer_state.json"))

            print("Saving substracted model")
            if trainer.is_deepspeed_enabled:
                model_state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
            else:
                trainer.model.to('cpu')
                model_state_dict = trainer.model.state_dict()
            task_vector = TaskVector(oracle_model.state_dict(), model_state_dict)
            task_vector = -task_vector
            model_sd = task_vector.apply_to(oracle_model, scaling_coef=cfg.tv_scaling_coef, in_place=False)
            sub_save_dir = os.path.join(cfg.save_dir, 'substracted_model')
            trainer._save(sub_save_dir, model_sd)

    if cfg.forget_loss == "CTV":
        print("Saving CTV mask")
        mask_save_dir = os.path.join(cfg.save_dir, 'CTV_mask')
        os.makedirs(mask_save_dir, exist_ok=True)
        mask = {key: (tensor.cpu() > trainer.mask_threshold[key]).bool() for key, tensor in trainer.m_vector.items()}
        if mask_previous is not None:
            mask = {key: mask[key] | mask_previous[key].cpu() for key in mask.keys()}
        torch.save(mask, os.path.join(mask_save_dir, "CTV_mask.pt"))
        torch.save(trainer.mask_threshold, os.path.join(mask_save_dir, "mask_threshold.pt"))


if __name__ == "__main__":
    main()

