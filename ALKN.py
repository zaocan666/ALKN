import os
import shutil
import sys
import time
from typing import Optional
import tqdm
import math
import numpy as np
import torch
from torch import nn
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import hp_params
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer import TRAINER_STATE_NAME, logger
from transformers.trainer_callback import TrainerState
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_torch_xla_available
)
from transformers.training_args import ParallelMode, OptimizerNames
from transformers.trainer import _is_peft_model, is_accelerate_available
from accelerate import skip_first_batches
from accelerate.utils import DistributedType
import torch.distributed as dist
if is_apex_available():
    from apex import amp
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

from scipy.stats import norm
from data_module import get_batch_loss, convert_raw_data_to_model_format

class ALKN_trainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.initial_model = kwargs.pop('oracle_model')
        self.forget_loss = kwargs.pop('forget_loss')
        self.unmask_ratio_start = kwargs.pop('unmask_ratio')
        self.unmask_ratio_end = self.unmask_ratio_start/2
        self.unmask_ratio_decrements = 5
        self.unmask_ratio = self.unmask_ratio_start
        self.threshold_update_step = kwargs.pop('threshold_update_step')
        self.mask_previous = kwargs.pop('mask_previous')
        self.mask_threshold_previous = kwargs.pop('mask_threshold')
        self.mask_threshold = self.mask_threshold_previous
        self.low_lr_ratio = kwargs.pop('low_lr_ratio')
        self.mask_p_k = kwargs.pop('mask_p_k') 
        self.init_prob_k = kwargs.pop('init_prob_k')
        super(ALKN_trainer, self).__init__(*args, **kwargs)
        
        
        if self.forget_loss == "CTV":
            self.get_retain_gradient()
            with torch.no_grad():
                self.get_forget_probability()
        self.accelerator.free_memory()

    def get_retain_gradient(self):
        retain_gradient = {}
        m_vector = {}
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)
        model.train()
        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        print("Calculating retain gradient")
        it = 0
        for inputs in tqdm.tqdm(train_dataloader):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                _, retain_inputs = inputs
                retain_input_ids, retain_labels, retain_attention_mask, _ = retain_inputs
                retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
            
            if self.args.n_gpu > 1:
                retain_loss = retain_loss.mean()

            self.accelerator.backward(retain_loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in retain_gradient:
                        retain_gradient[key] = tensor.grad.data
                    else:
                        retain_gradient[key] += tensor.grad.data
            it += 1
            model.zero_grad()

        for key, tensor in model.named_parameters():
            retain_gradient[key] = retain_gradient[key]/it

        self.retain_gradient = retain_gradient

        self.m_vector = {}
        for key in self.retain_gradient.keys():
            self.m_vector[key] = torch.zeros_like(self.retain_gradient[key])
        
        if self.mask_previous != None:
            for key in self.mask_previous.keys():
                self.mask_previous[key] = self.mask_previous[key].to(self.retain_gradient[key].device)

    def get_forget_probability(self):
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)
        model.train()
        model = self.accelerator.prepare(model)

        if model is not self.model:
            self.model_wrapped = model

        print("Calculating forget probability")

        forget_init_probs = {}
        for inputs in tqdm.tqdm(train_dataloader):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                forget_inputs, _ = inputs
                input_ids, labels, attention_mask, idx = forget_inputs
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1) # shape [16, 500, 51200]

            shifted_labels = labels[..., 1:].contiguous() # shape [16, 499]
            part_probs = probs[..., :-1, :].contiguous() # shape [16, 499, 51200]
            
            shifted_labels_safe = shifted_labels.clone()
            shifted_labels_safe[shifted_labels_safe == -100] = 0

            shifted_labels_safe = shifted_labels_safe.unsqueeze(-1) # shape [16, 499, 1]
            selected_probs = torch.gather(part_probs, -1, shifted_labels_safe) # shape [16, 499, 1]
            selected_probs = selected_probs.squeeze(-1) # shape [16, 499]

            # 将shifted_labels中值为-100的位置在selected_logits中设为0
            selected_probs[shifted_labels == -100] = 0
            for i, id in enumerate(idx):
                forget_init_probs[id.item()] = selected_probs[i]
        
        self.forget_init_probs = torch.zeros([len(self.train_dataset), selected_probs.shape[1]], device=selected_probs.device)
        for i in range(len(self.train_dataset)):
            self.forget_init_probs[i] = forget_init_probs[i]
        
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.forget_loss == "CTV":
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask, idx = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            
            shifted_labels = labels[..., 1:].contiguous() # [16, 499]
            pred = outputs.logits[..., :-1, :].contiguous() # shape [16, 499, 51200]
            pred = pred.log_softmax(dim=-1)

            with torch.no_grad():
                shifted_labels_safe = shifted_labels.clone()
                shifted_labels_safe[shifted_labels_safe == -100] = 0

                init_probs = self.forget_init_probs[idx].clone()
                midpoint = torch.Tensor([0.8]).to("cuda")
                init_probs = torch.sigmoid((init_probs-midpoint)*self.init_prob_k)-torch.sigmoid(-midpoint*self.init_prob_k)
                init_probs = torch.clamp(init_probs, max=1)

                true_dist = torch.zeros_like(pred)

                true_dist.scatter_(2, shifted_labels_safe.unsqueeze(2), init_probs.unsqueeze(2))
                true_dist = true_dist.detach()

            loss = torch.sum(-true_dist * pred, dim=2)
            loss = torch.sum(loss*(shifted_labels != -100).type(true_dist.dtype))/(shifted_labels != -100).sum()

        return (loss, outputs) if return_outputs else loss

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    if self.forget_loss == "CTV":
                        with torch.no_grad():
                            self.mask_gradient(model)

                    if self.forget_loss == "CTV":
                        # stop_tune_step = max_steps // 2
                        stop_tune_step = max_steps
                        if not self.mask_threshold_previous:
                            if ((self.state.global_step+1) % max(1, stop_tune_step // self.unmask_ratio_decrements) == 0) and (self.state.global_step+1) < stop_tune_step:
                                # self.unmask_ratio 这样运算：self.unmask_ratio_start和self.unmask_ratio_end之间的等差数列，共self.unmask_ratio_decrements个数，根据self.state.global_step与max_steps的比值确定当前的unmask_ratio   
                                self.unmask_ratio = self.unmask_ratio_start - (self.unmask_ratio_start - self.unmask_ratio_end) * (self.state.global_step+1) / stop_tune_step
                                self.log({'unmask_ratio': self.unmask_ratio})

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def mask_gradient(self, model):
        for key, tensor in model.named_parameters():
            # tensor.grad.data = tensor.grad.data * self.retain_gradient[key]
            mul = tensor.grad.data * self.retain_gradient[key]
            self.m_vector[key] += mul
            std = mul.std()
            if self.mask_previous != None:
                self.m_vector[key] += (-std * self.mask_p_k * self.mask_previous[key].type(tensor.grad.data.dtype))

        if (self.state.global_step % self.threshold_update_step == 0) and (not self.mask_threshold_previous):
            self.mask_threshold = {}
            trim_ratio = 0.15
            assert self.unmask_ratio_end > trim_ratio
            for key, tensor in model.named_parameters():
                if self.unmask_ratio == 1:
                    self.mask_threshold[key] = torch.min(self.m_vector[key]).item()-1
                    continue

                k = max(5, int(self.m_vector[key].numel() * trim_ratio))
                max_m = torch.topk(self.m_vector[key].flatten(), k=k).values[-1].item()
                min_m = torch.topk(self.m_vector[key].flatten(), k=k, largest=False).values[-1].item()
                adjusted_percent = (self.unmask_ratio*self.m_vector[key].numel() - k+1) / (self.m_vector[key].numel() - 2 * k + 2)
                self.mask_threshold[key] = max_m - (max_m - min_m) * adjusted_percent

        parameter_sum = 0
        unmask_sum = 0
        overlap_unmask_sum = 0
        for key, tensor in model.named_parameters():
            tensor.grad.data = tensor.grad.data * (self.m_vector[key] > self.mask_threshold[key]).type(tensor.grad.data.dtype)
            parameter_sum += tensor.grad.data.numel()
            unmask_sum += torch.sum(self.m_vector[key] > self.mask_threshold[key])
            if self.mask_previous != None:
                # tensor.grad.data[self.mask_previous[key]] *= self.low_lr_ratio
                overlap_unmask_sum += torch.sum((self.m_vector[key] > self.mask_threshold[key]) & self.mask_previous[key])
        self.log({'actual_unmask_ratio': (unmask_sum / parameter_sum).item()})
        self.log({'overlap_unmask_ratio': (overlap_unmask_sum / (unmask_sum+1e-7)).item()})
        return

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:

            if self.mask_previous != None:
                mask_ratio = {}
                for k,v in self.mask_previous.items():
                    ratio = v.sum()/v.numel()
                    mask_ratio[k] = ratio

                sorted_items = sorted(mask_ratio.items(), key=lambda x: x[1])
                cut_index = int(len(sorted_items)*0.5)
                low_ratio_keys = [k for k, v in sorted_items[:cut_index]]
                high_ratio_keys = [k for k, v in sorted_items[cut_index:]]
            else:
                low_ratio_keys = []
                for name, _ in opt_model.named_parameters():
                    low_ratio_keys.append('module.'+name)

            optimizer_grouped_parameters = []
            decay_parameters = self.get_decay_parameter_names(opt_model)
            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                mask_name = 'module.'+name
                if mask_name in low_ratio_keys:
                    lr = self.args.learning_rate
                elif mask_name in high_ratio_keys:
                    lr = self.args.learning_rate*self.low_lr_ratio
                else:
                    raise ValueError(f"mask_name {mask_name} not in low_ratio_keys or high_ratio_keys")

                if name in decay_parameters:
                    weight_decay = self.args.weight_decay
                else:
                    weight_decay = 0.0

                optimizer_grouped_parameters.append({"params": param, "lr": lr, "weight_decay": weight_decay})  # Mask 为 1 的部分

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    def _get_learning_rate(self):
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]
            else:
                # last_lr = self.lr_scheduler.get_last_lr()[0]
                last_lr = max(self.lr_scheduler.get_last_lr())
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr