CUDA_VISIBLE_DEVICES=1 python run_eval.py \
  --epoch 9 \
  --vicuna_hf_repo jsun39/kidspeak_vicuna \
  --delta_ckpt_base_path ../checkpoints/kidspeak_small/check \
  --test_file ../dataset/output/talkbank_v1_3_enni_post_test.json \
  --save_file_path ../result/kidspeak_small/enni/epoch_9.txt \
  --base_audio_dir ../data \
  --whisper_pretrained small
