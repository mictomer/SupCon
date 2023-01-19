import argparse
import logging
import os
import time
import torch
import yaml
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

from SupCon.tools import utils


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="configs/train/train_supcon_resnet18_cifar10_stage1.yml")
    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


def train(p_supcon, lSet_indices):
    backbone = p_supcon["model"]["backbone"]
    ckpt_pretrained = p_supcon['model']['ckpt_pretrained']
    num_classes = p_supcon['model']['num_classes']
    amp = p_supcon['train']['amp']
    ema = p_supcon['train']['ema']
    ema_decay_per_epoch = p_supcon['train']['ema_decay_per_epoch']
    n_epochs = p_supcon["train"]["n_epochs"]
    logging_name = p_supcon['train']['logging_name']
    target_metric = p_supcon['train']['target_metric']
    stage = p_supcon['train']['stage']
    data_dir = p_supcon["dataset"]
    optimizer_params = p_supcon["optimizer"]
    scheduler_params = p_supcon["scheduler"]
    criterion_params = p_supcon["criterion"]

    batch_sizes = {
        "train_batch_size": p_supcon["dataloaders"]["train_batch_size"],
        'valid_batch_size': p_supcon['dataloaders']['valid_batch_size']
    }
    num_workers = p_supcon["dataloaders"]["num_workers"]

    utils.seed_everything()

    # create model, loaders, optimizer, etc
    transforms = utils.build_transforms(second_stage=(stage == 'second'))
    loaders = utils.build_loaders(data_dir, transforms, batch_sizes, num_workers, lSet_indices, second_stage=(stage == 'second'))
    model = utils.build_model(backbone, second_stage=(stage == 'second'), num_classes=num_classes, ckpt_pretrained=ckpt_pretrained).cuda()

    if ema:
        iters = len(loaders['train_features_loader'])
        ema_decay = ema_decay_per_epoch**(1/iters)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    optim = utils.build_optim(model, optimizer_params, scheduler_params, criterion_params)
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )

    # handle logging (regular logs, tensorboard, and weights)
    if logging_name is None:
        logging_name = "stage_{}_model_{}_dataset_{}".format(stage, backbone, data_dir.split("/")[-1])

    shutil.rmtree("weights/{}".format(logging_name), ignore_errors=True)
    shutil.rmtree(
        "runs/{}".format(logging_name),
        ignore_errors=True,
    )
    shutil.rmtree(
        "logs/{}".format(logging_name),
        ignore_errors=True,
    )
    os.makedirs(
        "logs/{}".format(logging_name),
        exist_ok=True,
    )

    writer = SummaryWriter("runs/{}".format(logging_name))
    logging_dir = "logs/{}".format(logging_name)
    logging_path = os.path.join(logging_dir, "train.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")

    # epoch loop
    metric_best = 0
    best_model = model

    scaler = torch.cuda.amp.GradScaler()
    if not amp:
        scaler = None

    for epoch in range(n_epochs):
        utils.add_to_logs(logging, "{}, epoch {}".format(time.ctime(), epoch))

        start_training_time = time.time()
        if stage == 'first':
            train_metrics = utils.train_epoch_constructive(loaders['train_supcon_loader'], model, criterion, optimizer, scaler, ema)
        else:
            train_metrics = utils.train_epoch_ce(loaders['train_features_loader'], model, criterion, optimizer, scaler, ema)
        end_training_time = time.time()

        if ema:
            copy_of_model_parameters = utils.copy_parameters_from_model(model)
            ema.copy_to(model.parameters())

        start_validation_time = time.time()
        if stage == 'first':
            valid_metrics_projection_head = utils.validation_constructive(loaders['valid_loader'], loaders['train_features_loader'], model, scaler)
            model.use_projection_head(False)
            valid_metrics_encoder = utils.validation_constructive(loaders['valid_loader'], loaders['train_features_loader'], model, scaler)
            model.use_projection_head(True)
            print(
                'epoch {}/{}, train time {:.2f} valid time {:.2f} train loss {:.2f}\nvalid acc dict projection head {}\nvalid acc dict encoder {}'.format(
                    epoch,
                    n_epochs,
                    end_training_time - start_training_time,
                    time.time() - start_validation_time,
                    train_metrics['loss'], valid_metrics_projection_head, valid_metrics_encoder))
            valid_metrics = valid_metrics_projection_head
        else:
            valid_metrics = utils.validation_ce(model, criterion, loaders['valid_loader'], scaler)
            print(
                'epoch {}/{}, train time {:.2f} valid time {:.2f} train loss {:.2f}\n valid acc dict {}\n'.format(
                    epoch,
                    n_epochs,
                    end_training_time - start_training_time,
                    time.time() - start_validation_time,
                    train_metrics['loss'], valid_metrics))


        # write train and valid metrics to the logs
        utils.add_to_tensorboard_logs(writer, train_metrics['loss'], "Loss/train", epoch)
        for valid_metric in valid_metrics:
            try:
                utils.add_to_tensorboard_logs(writer, valid_metrics[valid_metric],
                                              '{}/validation'.format(valid_metric), epoch)
            except AssertionError:
                # in case valid metric is a list
                pass

        if stage == 'first':
            utils.add_to_logs(
                logging,
                "Epoch {}, train loss: {:.4f}\nvalid metrics projection head: {}\nvalid metric encoder: {}".format(
                    epoch,
                    train_metrics['loss'],
                    valid_metrics_projection_head,
                    valid_metrics_encoder
                ),
            )
        else:
            utils.add_to_logs(
                logging,
                "Epoch {}, train loss: {:.4f} valid metrics: {}".format(
                    epoch,
                    train_metrics['loss'],
                    valid_metrics
                ),
            )
        # check if the best value of metric changed. If so -> save the model
        if valid_metrics[target_metric] > metric_best:
            utils.add_to_logs(
                logging,
                "{} increased ({:.6f} --> {:.6f}).  Saving model ...".format(target_metric,
                    metric_best, valid_metrics[target_metric]
                ),
            )

            os.makedirs(
                "weights/{}".format(logging_name),
                exist_ok=True,
            )

            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()},
                "weights/{}/epoch{}".format(logging_name, epoch),
            )
            metric_best = valid_metrics[target_metric]
            best_model = model

        # if ema is used, go back to regular weights without ema
        if ema:
            utils.copy_parameters_to_model(copy_of_model_parameters, model)

        scheduler.step()

    writer.close()

    p_supcon['supcon_model_path'] = os.path.join('results/{}'.format(p_supcon['dataset'].split('/')[1]), f'supcon_checkpoint.pth.tar')
    torch.save(best_model.state_dict(), p_supcon['supcon_model_path'])
