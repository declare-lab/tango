CUDA_VISIBLE_DEVICES=0 python inference.py --original_args="saved/1681728144/summary.jsonl" \
--model="saved/1681728144/epoch_39/pytorch_model_2.bin" --num_steps 200 --guidance 3 --num_samples 1