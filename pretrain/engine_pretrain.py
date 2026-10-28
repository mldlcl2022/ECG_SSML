import math
import sys
from typing import Iterable

import torch
from torch.amp import autocast

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch_con(model: torch.nn.Module,
                        data_loader: Iterable,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        loss_scaler,
                        log_writer=None,
                        config=None,
                        loss_image=None,
                        loss_signal=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = config['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (images, signals) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        images = images.to(device, non_blocking=True)
        signals = signals.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            logits_image, logits_signal = model(images, signals)

            labels = torch.arange(len(images)).to(device)
            image_loss = loss_image(logits_image, labels)
            signal_loss = loss_signal(logits_signal, labels)
            loss = (image_loss + signal_loss) / 2

        loss_value = loss.item()
        image_loss_value = image_loss.item()
        signal_loss_value = signal_loss.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, n=images.shape[0])

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        image_loss_value_reduce = misc.all_reduce_mean(image_loss_value)
        signal_loss_value_reduce = misc.all_reduce_mean(signal_loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('pretrain_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('image_loss', image_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('signal_loss', signal_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_gen(model: torch.nn.Module,
                        data_loader: Iterable,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        loss_scaler,
                        log_writer=None,
                        config=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = config['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (images, signals) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        images = images.to(device, non_blocking=True)
        signals = signals.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            loss, loss_img, loss_sig, _ = model(images, signals)

        loss_value = loss.item()
        loss_img_value = loss_img.item()
        loss_sig_value = loss_sig.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, n=images.shape[0])

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_img_value_reduce = misc.all_reduce_mean(loss_img_value)
        loss_sig_value_reduce = misc.all_reduce_mean(loss_sig_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('pretrain_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('loss_img', loss_img_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_sig', loss_sig_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_con(model: torch.nn.Module,
                 data_loader: Iterable,
                 device: torch.device,
                 loss_image=None,
                 loss_signal=None):
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, signals in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        signals = signals.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            if signals.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_image_list = []
                logits_signal_list = []
                for i in range(signals.size(1)):
                    logits_image, logits_signal = model(images, signals[:, i])
                    logits_image_list.append(logits_image)
                    logits_signal_list.append(logits_signal)

                logits_image_stack = torch.stack(logits_image_list, dim=1)  # [B, n_drops, B]
                logits_signal_stack = torch.stack(logits_signal_list, dim=1)

                logits_image = logits_image_stack.mean(dim=1)   # [B, B]
                logits_signal = logits_signal_stack.mean(dim=1) # [B, B]
            else:
                logits_image, logits_signal = model(images, signals)

            labels = torch.arange(len(images)).to(device)
            image_loss = loss_image(logits_image, labels)
            signal_loss = loss_signal(logits_signal, labels)
            loss = (image_loss + signal_loss) / 2

        metric_logger.meters['loss'].update(loss.item())

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return valid_stats

@torch.no_grad()
def evaluate_gen(model: torch.nn.Module,
                 data_loader: Iterable,
                 device: torch.device):
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, signals in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        signals = signals.to(device, non_blocking=True)
        
        with autocast(device_type='cuda'):
            if signals.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                loss_list = []
                for i in range(signals.size(1)):
                    loss, *_ = model(images, signals[:, i])  
                    loss_list.append(loss)
                loss = torch.stack(loss_list, dim=0).mean()
            else:
                loss, *_ = model(images, signals)

        metric_logger.meters['loss'].update(loss.item())

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return valid_stats