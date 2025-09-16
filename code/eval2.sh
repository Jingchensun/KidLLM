CUDA_VISIBLE_DEVICES=2 python run_eval.py \
  --epoch 9 \
  --vicuna_hf_repo jsun39/kidspeak_vicuna \
  --delta_ckpt_base_path ../checkpoints/kidspeak_small/check \
  --test_file ../dataset/output/english_children_test.json \
  --save_file_path ../result/kidspeak_small/english_children/epoch_9.txt \
  --base_audio_dir ../data \
  --whisper_pretrained small
