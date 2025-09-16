CUDA_VISIBLE_DEVICES=3 python run_eval.py \
  --epoch 9 \
  --vicuna_hf_repo jsun39/kidspeak_vicuna \
  --delta_ckpt_base_path ../checkpoints/kidspeak_small/check \
  --test_file ../dataset/output/ultrasuite_disorder_test.json \
  --save_file_path ../result/kidspeak_small/ultrasuite_disorder/epoch_9.txt \
  --base_audio_dir ../data \
  --whisper_pretrained small
