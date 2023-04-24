# Train the LDM from scracth with a flan-t5-large text encoder
accelerate launch train.py \
--train_file="data/train_audiocaps.json" --validation_file="data/valid_audiocaps.json" --test_file="data/test_audiocaps_subset.json" \
--text_encoder_name="google/flan-t5-large" --scheduler_name="stabilityai/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config.json" --freeze_text_encoder \
--gradient_accumulation_steps 4 --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --augment \
--learning_rate=3e-5 --num_train_epochs 40 --snr_gamma 5 \
--text_column captions --audio_column location --checkpointing_steps="best"

# Continue training the LDM from our checkpoint using the --hf_model argument
accelerate launch train.py \
--train_file="data/train_audiocaps.json" --validation_file="data/valid_audiocaps.json" --test_file="data/test_audiocaps_subset.json" \
--hf_model "declare-lab/tango" --unet_model_config="configs/diffusion_model_config.json" --freeze_text_encoder \
--gradient_accumulation_steps 4 --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --augment \
--learning_rate=3e-5 --num_train_epochs 40 --snr_gamma 5 \
--text_column captions --audio_column location --checkpointing_steps="best"