from model.header import *
from datasets import *
from model import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    # model configurations
    parser.add_argument('--image_root_path', type=str) # the directory that stores all images
    parser.add_argument('--vicuna_hf_repo', type=str, default='jsun39/kidspeak_vicuna', 
                       help='Hugging Face repository for vicuna model')
    parser.add_argument('--delta_ckpt_path', type=str) # the delta parameters trained in stage 1
    parser.add_argument('--whisper_pretrained', type=str, default='small', 
                       help='whisper model size: ours, tiny, base, small, medium, large')
    parser.add_argument('--whisper_path', type=str, default='', 
                       help='path to custom whisper checkpoint (used when whisper_pretrained=ours)')
    parser.add_argument('--max_tgt_len', type=int) # the maximum sequence length
    parser.add_argument('--stage', type=int) # the maximum sequence length
    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):
    config_env(args)
    #deepspeed config /home/shijie/ra/PandaGPT_expts/PandaGPT_whisper/code/dsconfig/openllama_peft_stage_1.json
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    
    # {'audio_paths': 'KIDS/talkbank_dataset/official_v1.2/usable/FASA_Clinical_Other/out/05BG/216.867_217.627.mp3',
    #  'output_texts': [{'from': 'human', 'value': "Engage with the audio and articulate your impressions of the speaker's voice."}, 
    # {'from': 'gpt', 'value': 'It sounds like a child is articulating words in the clip.'}, {'from': 'human', 'value': 'Would you kindly write down the content of the speech?'},
    # {'from': 'gpt', 'value': 'This is the english transcription, do you want more colors'},
    # {'from': 'human', 'value': 'Notice any disruptions in fluency or rhythm of speech that may hint at a speech hindrance.'},
    # {'from': 'gpt', 'value': 'Expression is clear.'}]}
    train_data, train_iter, sampler = load_sft_dataset(args)
    val_data, val_iter, val_sampler = load_sft_dataset_val(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    val_length = len(val_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
 
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total=length)    # maximum total number
    val_pbar = tqdm(total=val_length)
    current_step = 0
    current_val_step = 0
    best = 0
    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            current_step += 1
        torch.distributed.barrier()
        val_accs = []
        # save at the end of the training
        torch.distributed.barrier()
        agent.save_model(args['save_path'], epoch_i)
 
        # for val_batch in val_iter:
        #     val_acc = agent.eval_model(
        #         val_batch, 
        #         current_step=current_val_step, 
        #         pbar=val_pbar
        #     )
        #     current_val_step += 1
        #     val_accs.append(val_acc)
        # meanvalperf = sum(val_accs)/len(val_accs)
        # if meanvalperf > best:
        #     best = meanvalperf
        #     print('BEST MODEL SO FAR')
        #     # save at the end of the training
        #     torch.distributed.barrier()
        #     agent.save_model(args['save_path'], epoch_i)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
