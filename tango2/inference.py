import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tools.torch_tools import read_wav_file
from tango import Tango
import numpy as np
import librosa
import laion_clap


def clap_score_computation(wav_output_dir,text_prompts):
    
    
    
    
    cos_sim = torch.nn.CosineSimilarity()
    
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.
    model.eval()
    
    output_dir = wav_output_dir
    
    audio_file = [
        "{}/output_{}.wav".format(output_dir,i) for i in range(len(text_prompts))
    ]
    with torch.no_grad():
        audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=True).cpu()
        
    with torch.no_grad():
        text_embed = model.get_text_embedding(text_prompts, use_tensor=True).cpu()
    
    clap_score = torch.mean(cos_sim(audio_embed,text_embed)).item()
    return clap_score




class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    # Load Models #
    if train_args.hf_model:
        tango = Tango(train_args.hf_model, "cpu")
        vae, stft, model = tango.vae.cuda(), tango.stft.cuda(), tango.model.cuda()
    else:
        name = "audioldm-s-full"
        vae, stft = build_pretrained_models(name)
        vae, stft = vae.cuda(), stft.cuda()
        model = AudioDiffusion(
            train_args.text_encoder_name, train_args.scheduler_name, train_args.unet_model_name, train_args.unet_model_config, train_args.snr_gamma
        ).cuda()
        model.eval()
    
    # Load Trained Weight #
    device = vae.device()
    model.load_state_dict(torch.load(args.model))
    
    scheduler = DDPMScheduler.from_pretrained(train_args.scheduler_name, subfolder="scheduler")
    evaluator = EvaluationHelper(16000, "cuda:0")
    
    
    
    wandb.init(project="Text to Audio Diffusion Evaluation")

    
    # Load Data #
    if train_args.prefix:
        prefix = train_args.prefix
    else:
        prefix = ""
        
    text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    text_prompts = [prefix + inp for inp in text_prompts]
    
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
    
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        
        with torch.no_grad():
            latents = model.inference(text, scheduler, num_steps, guidance, num_samples, disable_progress=True)
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            all_outputs += [item for item in wave]
            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    
   
    output_dir = "outputs/{}_{}_steps_{}_guidance_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance)
    os.makedirs(output_dir, exist_ok=True)
    for j, wav in enumerate(all_outputs):
        sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)
    
    clap_score = clap_score_computation(output_dir,text_prompts)
    

    result = evaluator.main(output_dir, args.test_references)
    result["Steps"] = num_steps
    result["Guidance Scale"] = guidance
    result["Test Instances"] = len(text_prompts)
    result["Clap Score"] =  np.round(clap_score,2)
    wandb.log(result)
    
    result["scheduler_config"] = dict(scheduler.config)
    result["args"] = dict(vars(args))
    result["output_dir"] = output_dir

    with open("outputs/summary.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n\n")
            
    
        
if __name__ == "__main__":
    main()