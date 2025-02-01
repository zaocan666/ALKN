import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
# from evaluate_util import get_dataloader, get_all_evals
from evaluate_util_CL import run_generation, eval_rouge_recall
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        
        self.ref_policy = "fine_tuned"
        self.npo_coeff = kwargs.pop('npo_coeff')
        self.grad_diff_coeff = kwargs.pop('grad_diff_coeff')
        self.KL_coeff = kwargs.pop('KL_coeff')
        self.beta = kwargs.pop('beta')
        self.my_eval_dataloader = kwargs.pop('my_eval_dataloader')
        self.eval_threshold = kwargs.pop('eval_threshold')
        self.performance_threshold = kwargs.pop('performance_threshold')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        
        if self.loss_type == "KL" or "npo" in self.loss_type:
            if self.accelerator.state.deepspeed_plugin:
                self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
            else:
                self.oracle_model.to("cuda")
                self.oracle_model.eval()
                for param in self.oracle_model.parameters():
                    param.requires_grad = False

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask, _ = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
        
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask, _ = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        ### Implement the NPO
        elif self.loss_type == 'npo':
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

        elif self.loss_type == 'npo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask, _ = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss
            
        elif self.loss_type == 'npo_KL':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask, _ = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = self.npo_coeff * forget_loss + self.KL_coeff * retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask, _ = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask, _ = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type == "dpo":
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask, _ = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask, _ = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, labels)


            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            print(loss.item())
            loss = -pi_logratios.mean()
            loss = -idk_loss_current.mean()

            outputs = forget_outputs
        
        elif self.loss_type == "task_vector":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, _ = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            loss = forget_loss

        return (loss, outputs) if return_outputs else loss
        
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask, _ = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        if self.control.should_training_stop:
            return {}

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        if self.accelerator.is_local_main_process:
            print(f"start evaluation, step: {self.state.global_step}")
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg


        gen_outputs = []
        ground_truths = []
        all_indices = []

        eval_dataloader = self.accelerator.prepare(self.my_eval_dataloader)
        for batch in eval_dataloader:
            input_ids, labels, attention_mask, indices = batch
            all_indices.extend(indices.cpu().numpy().tolist())
            batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

            with torch.no_grad():
                input_string, gen_output, gt = run_generation(eval_cfg, batch, model, tokenizer=self.tokenizer)
                gen_outputs.extend(gen_output)
                ground_truths.extend(gt)

        self.accelerator.wait_for_everyone()
        eval_logs = eval_rouge_recall(gen_outputs, ground_truths, all_indices)
        avg_rouge = np.array(list(eval_logs['rougeL_recall'].values())).mean()
        eval_score = avg_rouge

        if self.accelerator.is_local_main_process:
            self.log({'rougeL_recall': eval_score})

        if eval_score <= self.performance_threshold:
            print(f"Performance threshold reached. Stopping training. eval_score: {eval_score}")
            self.control.should_training_stop = True  # 设置停止标志

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        idx = [s[3] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(idx)))
    return rets



def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


