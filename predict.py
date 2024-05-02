# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
import json
import torch
from tqdm import tqdm
import soundfile as sf
from models import AudioDiffusion, DDPMScheduler
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from cog import BasePredictor, Input, Path


MODEL_URL = "https://weights.replicate.delivery/default/declare-lab/tango.tar"
MODEL_CACHE = "tango_weights"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.models = {k: Tango(name=k) for k in ["tango2", "tango2-full"]}

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Quiet speech and then and airplane flying away",
        ),
        model: str = Input(
            description="choose a model",
            choices=[
                "tango2",
                "tango2-full",
            ],
            default="tango2",
        ),
        steps: int = Input(description="inference steps", default=100),
        guidance: float = Input(description="guidance scale", default=3),
    ) -> Path:
        """Run a single prediction on the model"""

        tango = self.models[model]
        audio = tango.generate(prompt, steps, guidance)
        out = "/tmp/output.wav"
        sf.write(out, audio, samplerate=16000)
        return Path(out)


class Tango:
    def __init__(self, name="tango2", path=MODEL_CACHE, device="cuda:0"):
        # weights are downloaded from f"https://huggingface.co/declare-lab/{name}/tree/main" and saved to MODEL_CACHE
        vae_config = json.load(open(f"{path}/{name}/vae_config.json"))
        stft_config = json.load(open(f"{path}/{name}/stft_config.json"))
        main_config = json.load(open(f"{path}/{name}/main_config.json"))

        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = AudioDiffusion(**main_config).to(device)

        vae_weights = torch.load(
            f"{path}/{name}/pytorch_model_vae.bin", map_location=device
        )
        stft_weights = torch.load(
            f"{path}/{name}/pytorch_model_stft.bin", map_location=device
        )
        main_weights = torch.load(
            f"{path}/{name}/pytorch_model_main.bin", map_location=device
        )

        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        self.vae.eval()
        self.stft.eval()
        self.model.eval()

        self.scheduler = DDPMScheduler.from_pretrained(
            main_config["scheduler_name"], subfolder="scheduler"
        )

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from a list."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """Generate audio for a single prompt string."""
        with torch.no_grad():
            latents = self.model.inference(
                [prompt],
                self.scheduler,
                steps,
                guidance,
                samples,
                disable_progress=disable_progress,
            )
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]

    def generate_for_batch(
        self,
        prompts,
        steps=100,
        guidance=3,
        samples=1,
        batch_size=8,
        disable_progress=True,
    ):
        """Generate audio for a list of prompt strings."""
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k : k + batch_size]
            with torch.no_grad():
                latents = self.model.inference(
                    batch,
                    self.scheduler,
                    steps,
                    guidance,
                    samples,
                    disable_progress=disable_progress,
                )
                mel = self.vae.decode_first_stage(latents)
                wave = self.vae.decode_to_waveform(mel)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))
