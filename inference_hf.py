import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tango import Tango


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
        "--checkpoint", type=str, default="declare-lab/tango",
        help="Tango huggingface checkpoint"
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
        "--device", type=str, default="cuda:0",
        help="Device to use for inference."
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
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    num_steps, guidance, batch_size = args.num_steps, args.guidance, args.batch_size
    checkpoint = args.checkpoint
    
    # Load Models #
    tango = Tango(checkpoint, args.device)
    vae, stft, model = tango.vae, tango.stft, tango.model
    
    scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")
    evaluator = EvaluationHelper(16000, "cuda:0")
    
    # Load Data #
    prefix = ""
    text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    text_prompts = [prefix + inp for inp in text_prompts]
        
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    output_dir = "outputs/{}_steps_{}_guidance_{}".format(exp_id, num_steps, guidance)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate #
    all_outputs = []
        
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        
        with torch.no_grad():
            latents = model.inference(text, scheduler, num_steps, guidance)
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            all_outputs += [item for item in wave]
    
    # Save #
    for j, wav in enumerate(all_outputs):
        sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)

    result = evaluator.main(output_dir, args.test_references)
    result["Steps"] = num_steps
    result["Guidance Scale"] = guidance
    result["Test Instances"] = len(text_prompts)

    result["scheduler_config"] = dict(scheduler.config)
    result["args"] = dict(vars(args))
    result["output_dir"] = output_dir

    with open("outputs/tango_checkpoint_summary.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n\n")
            
            
if __name__ == "__main__":
    main()