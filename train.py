# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import datetime
import time
import yaml
from tqdm import tqdm
import constants as const
import dataset
import fastflow
import utils
import numpy as np
import cv2 
import paddle
import paddle.nn as nn
import random
import warnings
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

paddle.set_device(const.device)


def set_seed(seed):
    '''
    set_seed
    '''
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PL_SEED_WORKERS"] = "0"


def build_train_data_loader_paddle(args, config):
    '''
    create train dataloader
    '''
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )

    return paddle.io.DataLoader(
        train_dataset,
        batch_size=const.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )


def build_test_data_loader_paddle(args, config):
    '''
    create test dataloader
    '''
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )

    return paddle.io.DataLoader(
        test_dataset,
        batch_size=const.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model_paddle(config):
    '''
    create fastflow model
    '''
    model = fastflow.FastFlow(
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        use_norm=config['use_norm'],
        momentum = config['momentum'],
        channels = config['channels'],
        scales = config['scales'],
        clamp = config['clamp'],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if not p.stop_gradient)
        )
    )

    return model

def build_tensorbord(args, exp_dir):
    '''
    create summary_writer for log
    '''
    log_dir = os.path.join(
        const.CHECKPOINT_DIR, exp_dir, '{}'.format(args.category), 'log')
    summary_writer = SummaryWriter(log_dir=log_dir)
    return summary_writer

def build_optimizer_paddle(model):
    '''
    create optimizer
    '''
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=const.CLIP_NORM)
    return paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=const.LR, weight_decay=const.WEIGHT_DECAY, grad_clip=clip
    )

def train_one_epoch(dataloader, model, optimizer, epoch, category, summary_writer, mylogger):
    '''
    train 1 epoch
    '''
    model.train()
    loss_meter = utils.AverageMeter()
    pbar = tqdm(dataloader)

    train_reader_cost = 0.0 ## cost time on loading train dataset
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    for step, (data, file) in enumerate(pbar):
        # forward
        train_reader_cost += time.time() - reader_start
        train_start = time.time()
        ret = model(data)
        loss = ret["loss"]
        summary_writer.add_scalar("train_loss/iter", loss.item(), step)
        # backward  
        loss.backward()
        # optimizer
        optimizer.step()
        optimizer.clear_grad()
        
        train_run_cost += time.time() - train_start
        total_samples += data.shape[0]
        # log
        loss_meter.update(loss.item())
        message = "{}  Epoch {} - iter:{}/{} - lr:{} - loss = {:.3f}/{:.3f} (last/avg) - avg_reader_cost: {:.5f} sec - avg_batch_cost: {:.5f} sec - avg_samples: {} - avg_ips: {:.5f} images/sec".format(
              category, epoch + 1, step + 1, len(pbar), optimizer.get_lr(), loss_meter.val, loss_meter.avg, train_reader_cost, train_reader_cost + train_run_cost, total_samples, total_samples / (train_reader_cost + train_run_cost)
            )
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        mylogger.print(message)
        pbar.set_postfix_str(message)
        reader_start = time.time()
    


def eval_once(dataloader, model, eval = True):
    '''
    function:eval on test dataset
    
    return:
        auroc_px: pixel_level auc
        auroc_sp: image_level auc
    '''
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    preds = []
    gts = []

    for data, targets in dataloader:
        targets = targets.cpu().numpy().astype(int)

        with paddle.no_grad():
            ret = model(data)

        outputs = ret["anomaly_map"].detach().cpu()
        outputs = outputs.numpy()

        preds.append(outputs)
        gts.append(targets)
    
    preds = np.concatenate(preds, axis = 0)
    gts = np.concatenate(gts, axis = 0)

    targets = gts
    outputs = preds
    for i in range(targets.shape[0]):
        gt_list_sp.append(np.max(targets[i]))
        pr_list_sp.append(np.max(outputs[i]))
        if eval:
            outputs[i] = gaussian_filter(outputs[i], sigma=6)

        gt_list_px.extend(targets[i].ravel())
        pr_list_px.extend(outputs[i].ravel())
        
    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    print("EVAL:  image-auc : {:.6f} pixel-auc:{:.6f}".format(auroc_sp, auroc_px))
    
    return auroc_px, auroc_sp


def train(args, exp_dir):
    '''
    function:train all epoches
    
    return:
        best_pixel_auroc: picel_level auc
        best_image_auroc: image_level auc
    '''
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, exp_dir, '{}'.format(args.category))
    os.makedirs(checkpoint_dir, exist_ok=True)
    mylogger = utils.Logger(checkpoint_dir)
    utils.save_config(checkpoint_dir)

    best_pixel_auroc, best_image_auroc = 0, 0
    max_result = -1
    max_result_indice = 0
    count = 0
    config = yaml.safe_load(open(args.config, "r"))
    model_paddle = build_model_paddle(config)
    optimizer_paddle = build_optimizer_paddle(model_paddle)
    summary_writer = build_tensorbord(args, exp_dir)
    print("Loading training data")
    st = time.time()
    train_dataloader_paddle = build_train_data_loader_paddle(args, config)
    test_dataloader_paddle = build_test_data_loader_paddle(args, config)
    print("Took", time.time() - st)
    
    print("Start training")
    start_time = time.time()
    for epoch in range(const.NUM_EPOCHS):
        ###train one epoch
        train_one_epoch(train_dataloader_paddle, model_paddle, optimizer_paddle, epoch, args.category, summary_writer, mylogger)
        ### eval
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            pixel_auroc, image_auroc = eval_once(test_dataloader_paddle, model_paddle)
            summary_writer.add_scalar("pixel_auroc/epoch", pixel_auroc, epoch)
            summary_writer.add_scalar("image_auroc/epoch", image_auroc, epoch)
            
            current_result = pixel_auroc * const.PIXEL_AUROC_RATIO + image_auroc * const.IMAGE_AUROC_RATIO
            ### early stopping
            if current_result >= max_result:
                if current_result == max_result and pixel_auroc < best_pixel_auroc:
                    count = 0
                else: ## save best param
                    max_result = current_result
                    max_result_indice = epoch
                    best_image_auroc = image_auroc
                    best_pixel_auroc = pixel_auroc
                    paddle.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model_paddle.state_dict(),
                            "optimizer_state_dict": optimizer_paddle.state_dict(),
                        },
                        os.path.join(checkpoint_dir, "best.pdparams"),
                    )
                    count = 0
            else:
                count += 1
        ### save param of last epoch      
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            paddle.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_paddle.state_dict(),
                    "optimizer_state_dict": optimizer_paddle.state_dict(),
                },
                os.path.join(checkpoint_dir, "last.pdparams"),
            )
        if epoch - max_result_indice >= const.PATIENCE:
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    del model_paddle, optimizer_paddle, train_dataloader_paddle, test_dataloader_paddle
    mylogger.reset()
    return best_pixel_auroc, best_image_auroc



def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    
    parser.add_argument(
        "-cfg", "--config", type=str, default='configs/resnet18.yaml', help="path to config file"
    )

    parser.add_argument(
        '-cat',
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        default='bottle',
        help="category name in mvtec",
    )

    parser.add_argument(
        "--exp_dir",default='', type=str, help="path to save checkpoint"
    )
    

    parser.add_argument('--test_epochs', default=const.EVAL_INTERVAL, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
    parser.add_argument('--data', default="data",
                        help='input folder of the models ')

    parser.add_argument('--epochs', default=const.NUM_EPOCHS, type=int,
                        help='number of epochs to train the model , (default: 256)')
    parser.add_argument('--batch_size', default=const.TRAIN_BATCH_SIZE, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize ('
                             'dafault: "64")')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    def print_result(pixel_auroc_list_all, image_auroc_list_all):
            mean_pixel_auroc = 0
            mean_image_auroc = 0
            for category in pixel_auroc_list_all.keys():
                pixel_auroc = pixel_auroc_list_all[category]
                image_auroc= image_auroc_list_all[category]
                mean_pixel_auroc += pixel_auroc
                mean_image_auroc += image_auroc
                
                print('{:15}  pixel auroc:{:.3f}    image auroc:{:.3f}'.format(category, pixel_auroc, image_auroc,))
            print('{:15}  pixel auroc:{:.4f}    image auroc:{:.4f}'.format('mean', mean_pixel_auroc/ len(pixel_auroc_list_all),  mean_image_auroc / len(image_auroc_list_all)))
        
            
    set_seed(0)
    args = parse_args()
    const.NUM_EPOCHS = args.epochs
    const.EVAL_INTERVAL = args.test_epochs
    const.TRAIN_BATCH_SIZE = args.batch_size

    ### all model params and logs will be saved in const.CHECKPOINT_DIR/exp_dir
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    if args.exp_dir == '':
        exp_dir = "exp_{}_{}".format(const.TRAIN_BATCH_SIZE, len(os.listdir(const.CHECKPOINT_DIR)))
    else:
        exp_dir = args.exp_dir
        
    all_category = const.MVTEC_CATEGORIES.copy()
    pixel_auroc_list_all = {}
    image_auroc_list_all = {}
    ## train
    pixel_auroc_list, image_auroc_list = train(args, exp_dir)
    ## print results
    pixel_auroc_list_all[args.category] = pixel_auroc_list
    image_auroc_list_all[args.category] = image_auroc_list
    print('save:{}'.format(exp_dir))
    print_result(pixel_auroc_list_all, image_auroc_list_all)
