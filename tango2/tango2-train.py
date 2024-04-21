import time
import argparse
import json
import logging
import math
import os
# from tqdm import tqdm
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HuggingfaceDataset
from tqdm.auto import tqdm
from itertools import combinations
import soundfile as sf
import diffusers
import transformers
import tools.torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models import build_pretrained_models, AudioDiffusion, DPOAudioDiffusion
from transformers import SchedulerType, get_scheduler
import copy


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument(
        "--train_file", type=str, default="data/train_audiocaps.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/valid_audiocaps.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="A csv or a json file containing the test data for generation."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_name", type=str, default=None,
        help="UNet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--beta_dpo", type=float, default=5000,
        help="Beta value for DPO",
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="Tango model identifier from huggingface: declare-lab/tango",
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--audio_w_column", type=str, default="chosen",
        help="The name of the column in the dpo dataset containing the winner audio paths.",
    )
    parser.add_argument(
        "--audio_l_column", type=str, default="rejected",
        help="The name of the column in the dpo dataset containing the loser audio paths.",
    )
    parser.add_argument(
        "--text_column_dpo", type=str, default="prompt",
        help="The name of the column in the dpo dataset containing the input texts.",
    )
    
    parser.add_argument(
        "--dpo_beta", type=float, default=2000,
        help="Beta value for dpo.",
    )
    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every", type=int, default=5,
        help="Save model after every how many epochs when checkpointing_steps is set to best."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--sft_first_epochs", type=int, default=0,
        help="Whether to perform sft on preference data first",
    )
    parser.add_argument("--dataset_dir",type=str,default="~/.cache/huggingface/datasets",
                        help="Location to store the downloaded audio")
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


class Text2AudioDataset(Dataset):
    def __init__(self, dataset, prefix, text_column, audio_column, num_examples=-1):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, text in zip(self.indices, self.audios, inputs):
            self.mapper[index] = [audio, text]

        if num_examples != -1:
            self.inputs, self.audios = self.inputs[:num_examples], self.audios[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3 = self.inputs[index], self.audios[index], self.indices[index]
        return s1, s2, s3

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
        
class DPOText2AudioDataset(Dataset):
    def __init__(self, dataset, prefix, text_column, audio_w_column, audio_l_column, num_examples=-1):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios_w = list(dataset[audio_w_column])
        self.audios_l = list(dataset[audio_l_column])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio_w, audio_l, text in zip(self.indices, self.audios_w,self.audios_l,inputs):
            self.mapper[index] = [audio_w, audio_l, text]

        if num_examples != -1:
            self.inputs, self.audios_w, self.audios_l = self.inputs[:num_examples], self.audios_w[:num_examples], self.audios_l[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4 = self.inputs[index], self.audios_w[index], self.audios_l[index],self.indices[index]
        return s1, s2, s3, s4

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
        


        

def main():
    args = parse_args()
    accelerator_log_kwargs = {}

  

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()


    dataset = load_dataset('declare-lab/audio_alpaca')
    dataset_dir = args.dataset_dir
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project="Text to Audio Diffusion with DPO")
        
        
        logger.info(f"***** Writing audio file to {dataset_dir}/audio_alpaca *****")
        if not os.path.exists(f"{dataset_dir}/audio_alpaca"):
            os.makedirs(f"{dataset_dir}/audio_alpaca")
            
        for i in range(len(dataset['train'])):
            sf.write(f"{dataset_dir}/audio_alpaca/chosen_{i}.wav", dataset['train'][i]['chosen']['array'], samplerate=16000) 
            sf.write(f"{dataset_dir}/audio_alpaca/reject_{i}.wav", dataset['train'][i]['rejected']['array'], samplerate=16000)
        
        

    accelerator.wait_for_everyone()
    
    # Get the datasets
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        if args.validation_file is not None:
            data_files["test"] = args.validation_file

    extension = args.train_file.split(".")[-1]
    
    
    #data_files_train = data_files["train"]
    data_files.pop('train')
    #raw_dataset_train = load_dataset(extension, data_files=data_files_train)
    raw_dataset_valid_test = load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column
    
    prompt_list = prompt_list = dataset['train']['prompt']
    chosen_audio_path = [f"{dataset_dir}/audio_alpaca/chosen_{i}.wav" for i in range(len(dataset['train']))]
    rejected_audio_path = [f"{dataset_dir}/audio_alpaca/reject_{i}.wav" for i in range(len(dataset['train']))]
    raw_dataset_train = HuggingfaceDataset.from_pandas(pd.DataFrame({'prompt':prompt_list,
                                      'chosen':chosen_audio_path,
                                      'rejected':rejected_audio_path}))
   
    
 
    

    # Initialize models
    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    stft.eval()

   
    model = DPOAudioDiffusion(
    args.text_encoder_name, args.scheduler_name, args.unet_model_name, args.unet_model_config, args.snr_gamma, args.freeze_text_encoder, args.uncondition, args.beta_dpo
    )
        


    if args.hf_model:
        hf_model_path = snapshot_download(repo_id=args.hf_model)
        model.load_state_dict(torch.load("{}/pytorch_model_main.bin".format(hf_model_path), map_location="cpu"))
        accelerator.print("Successfully loaded checkpoint from:", args.hf_model)
        
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""
        
    
    
        
    with accelerator.main_process_first():
        
        train_dataset = DPOText2AudioDataset(raw_dataset_train, prefix, 'prompt', 'chosen', 'rejected', args.num_examples)
        eval_dataset = Text2AudioDataset(raw_dataset_valid_test["validation"], prefix, text_column, audio_column, args.num_examples)
        test_dataset = Text2AudioDataset(raw_dataset_valid_test["test"], prefix, text_column, audio_column, args.num_examples)

    
        accelerator.print("Num instances in train: {}, validation: {}, test: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances(), test_dataset.get_num_instances()))
        
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=eval_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=test_dataset.collate_fn)
    
   
    ## Initialize reference Unet
    ref_unet = copy.deepcopy(model.unet)
    ref_unet.requires_grad_(False)
    ref_unet.to(accelerator.device)
        
    if args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()
        if args.unet_model_config:
            optimizer_parameters = model.unet.parameters()
            accelerator.print("Optimizing UNet parameters.")
        else:
            optimizer_parameters = list(model.unet.parameters()) + list(model.group_in.parameters()) + list(model.group_out.parameters())
            accelerator.print("Optimizing UNet and channel transformer parameters.")
    else:
        optimizer_parameters = model.parameters()
        accelerator.print("Optimizing Text Encoder and UNet parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in ref_unet.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))
    # Optimizer 
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs+args.sft_first_epochs) * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info("*****  Running DPO *****")
    logger.info(f" Learning rate = {args.learning_rate}")
    logger.info(f" Number of warmup steps = {args.num_warmup_steps}")
    logger.info(f" DPO beta value = {args.beta_dpo}")
    logger.info(f" Doing {args.sft_first_epochs} epoch of SFT first")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            # path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
    # Duration of the audio clips in seconds
    duration, best_loss = 10, np.inf
    
                
    sft_epochs = args.sft_first_epochs
   
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            ## Each batch contains the prompt, chosen audio, reject audio

            with accelerator.accumulate(model):
                device = accelerator.device
                text, audio_w, audio_l, _ = batch
                target_length = int(duration * 102.4)

                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    mel_w, _, waveform_w = torch_tools.wav_to_fbank(audio_w, target_length, stft)
                    mel_l, _, waveform_l = torch_tools.wav_to_fbank(audio_l, target_length, stft)
                    mel_w = mel_w.unsqueeze(1).to(device)
                    mel_l = mel_l.unsqueeze(1).to(device)
                    
                    
                    #true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))
                    latent_w =  unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel_w))
                    latent_l =  unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel_l))
                    
                if sft_epochs > 0: 
                    ## Perform SFT on the prompt and preferred audio if sft_epochs > 0
                    
                    latents = latent_w
                    loss = model(latents, text, ref_unet, validation_mode=False , sft = True)
                    
                else:
                    
                    latents = torch.cat((latent_w,latent_l),dim=0)
                    loss = model(latents, text, ref_unet, validation_mode=False , sft = False)
                        
                    
                
            
    
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if (completed_steps+1) % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps+1 }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        


        model.eval()
        model.uncondition = False
        

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            
            with accelerator.accumulate(model) and torch.no_grad():
                device = model.device
                text, audios, _ = batch
                target_length = int(duration * 102.4)

                unwrapped_vae = accelerator.unwrap_model(vae)
                mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
                mel = mel.unsqueeze(1).to(device)
                true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

                val_loss = accelerator.unwrap_model(model).diffusion_forward(true_latent, text, validation_mode=True)
                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        model.uncondition = args.uncondition

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
            
            sft_epochs -= 1
            wandb.log(result)

            result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(epoch, result["train_loss"], result["val_loss"])
            
            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False
    
        
        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(args.output_dir, "best"))
                
            if (epoch + 1) % args.save_every == 0 and sft_epochs < 0: ## Only save state after doing sft_first
                accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

        if accelerator.is_main_process and args.checkpointing_steps == "epoch" and sft_epochs < 0: ## Only save state after doing sft_first
            accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))
            
            
            
if __name__ == "__main__":
    main()
