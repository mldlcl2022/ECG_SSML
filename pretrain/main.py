import os
import argparse
import datetime
import json
import time

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from engine_pretrain import train_one_epoch_con, train_one_epoch_gen, evaluate_con, evaluate_gen
from dataset.pretrain import build_dataset, get_dataloader
from models.pretrain.contrastive import ECGCLIP
from models.pretrain.generative import CrossAttentionModel
from util.optimizer import get_optimizer_from_config
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def parse() -> dict:
    parser = argparse.ArgumentParser('ECG pre-training')

    parser.add_argument('--config_path',
                        default='',
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config

def main(config):
    misc.init_distributed_mode(config['ddp'])

    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])
    
    # seed 세팅
    seed = config['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ECG 데이터셋 (image-signal pair)
    dataset_train = build_dataset(config['dataset'], split='train')
    dataset_valid = build_dataset(config['dataset'], split='valid')
    
    data_loader_train = get_dataloader(dataset_train,
                                       is_distributed=config['ddp']['distributed'],
                                       mode='train',
                                       **config['dataloader'])
    data_loader_valid = get_dataloader(dataset_valid,
                                       is_distributed=config['ddp']['distributed'],
                                       dist_eval=config['train']['dist_eval'],
                                       mode='eval',
                                       **config['dataloader'])

    if misc.is_main_process() and config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None
        
    # 모델 정의
    if config['model']['type'] == 'contrastive':
        model = ECGCLIP(image_model='vit_recon',
                        signal_model='st_mem')
    elif config['model']['type'] == 'generative':
        model = CrossAttentionModel(depth=12,
                                    decoder_depth=4,
                                    cross_attention_depth=1)
    
    model.to(device)
    model_without_ddp = model
    print(f"Model = {model_without_ddp}")
    
    '''
    Training 세팅
    '''
    # batch size 및 learning rate 설정
    eff_batch_size = config['dataloader']['batch_size'] * config['train']['accum_iter'] * misc.get_world_size()
    if config['train']['lr'] is None:
        config['train']['lr'] = config['train']['blr'] * eff_batch_size / 256

    print(f"base lr: {config['train']['lr'] * 256 / eff_batch_size}")
    print(f"actual lr: {config['train']['lr']}")
    print(f"accumulate grad iterations: {config['train']['accum_iter']}")
    print(f"effective batch size: {eff_batch_size}")

    if config['ddp']['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[config['ddp']['gpu']])
        model_without_ddp = model.module
    
    # optimizer 설정
    optimizer = get_optimizer_from_config(config['train'], model_without_ddp)
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(config, model_without_ddp, optimizer, loss_scaler)
    
    if config['model']['type'] == 'contrastive':
        train_one_epoch = train_one_epoch_con
        evaluate = evaluate_con
        loss_image = nn.CrossEntropyLoss()
        loss_signal = nn.CrossEntropyLoss()
        extra_kwargs = {'loss_image': loss_image, 'loss_signal': loss_signal}
    elif config['model']['type'] == 'generative':
        train_one_epoch = train_one_epoch_gen
        evaluate = evaluate_gen
        extra_kwargs = {}

    best_loss = float('inf')
    
    '''
    Training 시작
    '''
    print(f"Start training for {config['train']['epochs']} epochs")
    start_time = time.time()
    for epoch in range(config['start_epoch'], config['train']['epochs']):
        if config['ddp']['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      log_writer,
                                      config['train'],
                                      **extra_kwargs)
        valid_stats = evaluate(model,
                               data_loader_valid,
                               device,
                               **extra_kwargs)
        
        curr_loss = valid_stats['loss']
        if output_dir and curr_loss < best_loss:
            best_loss = curr_loss
            misc.save_model(config,
                            os.path.join(output_dir, 'best-loss.pth'),
                            epoch,
                            model_without_ddp,
                            optimizer,
                            loss_scaler)

            misc.save_model(config, 
                            os.path.join(output_dir, 'best_signal_encoder.pth'), 
                            epoch, 
                            model_without_ddp.signal_encoder)
            misc.save_model(config, 
                            os.path.join(output_dir, 'best_image_encoder.pth'),  
                            epoch, 
                            model_without_ddp.image_encoder)
            print((f"Best loss updated: {best_loss} at epoch {epoch}"))
            
        if output_dir and (epoch % 10 == 0 or epoch + 1 == config['train']['epochs']):
            misc.save_model(config,
                            os.path.join(output_dir, f'checkpoint-{epoch}.pth'),
                            epoch,
                            model_without_ddp,
                            optimizer,
                            loss_scaler)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     }

        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), 'a', encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

if __name__ == "__main__":
    config = parse()
    main(config)