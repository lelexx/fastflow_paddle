import argparse
import os
import datetime
import time
import yaml
from ignite.contrib import metrics
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
import torch

from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

paddle.set_device('gpu:0')

def set_seed(seed = 42):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PL_SEED_WORKERS"] = f"0"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def build_train_data_loader_paddle(args, config):
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
    model = fastflow.FastFlow(
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if not p.stop_gradient)
        )
    )

    return model

def build_tensorbord(args, exp_dir):
    log_dir = os.path.join(
        const.CHECKPOINT_DIR, exp_dir, '{}'.format(args.category), 'log')
    summary_writer = SummaryWriter(log_dir=log_dir)
    return summary_writer

def build_optimizer_paddle(model):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.9)
    #clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    return paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=const.LR, weight_decay=const.WEIGHT_DECAY, grad_clip=clip
    )

def train_one_epoch(dataloader, model, optimizer, epoch, category, summary_writer, logger):
    model.train()
    loss_meter = utils.AverageMeter()
    pbar = tqdm(dataloader)

    train_reader_cost = 0.0
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
        if step != len(dataloader) - 1 and (step + 1) % const.GRADIENT_SUM != 0:
            continue
        
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
        logger.print(message)
        pbar.set_postfix_str(message)
        reader_start = time.time()
    

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)
def eval_once(dataloader, model, eval = True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    preds = []
    gts = []
    preds_1 = []
    for data, targets in dataloader:
        DEBUG = False
        targets = targets.cpu().numpy().astype(int)
        

        
        # data_flip_1 = paddle.flip(data, [2])
        # data_flip_2 = paddle.flip(data, [3])
        # data_flip_3 = paddle.flip(data, [2, 3])
        with paddle.no_grad():
            ret = model(data)
            # ret_flip_1 = model(data_flip_1)
            # ret_flip_2 = model(data_flip_2)
            # ret_flip_3 = model(data_flip_3)
            
        # outputs = ret["anomaly_map"].detach().cpu()
        # outputs_flip_1 = ret_flip_1["anomaly_map"].detach().cpu()
        # outputs_flip_2 = ret_flip_2["anomaly_map"].detach().cpu()
        # outputs_flip_3 = ret_flip_3["anomaly_map"].detach().cpu()
        # outputs = (0.4 *outputs + 0.2*paddle.flip(outputs_flip_1, [2]) + 0.2*paddle.flip(outputs_flip_2, [3])+ 0.2*paddle.flip(outputs_flip_3, [2,3])) / 1
        # outputs = outputs.numpy()
        
        outputs_1 = ret["anomaly_map_1"].detach().cpu()
        # outputs_flip_1_1 = ret_flip_1["anomaly_map_1"].detach().cpu()
        # outputs_flip_2_1 = ret_flip_2["anomaly_map_1"].detach().cpu()
        # outputs_flip_3_1 = ret_flip_3["anomaly_map_1"].detach().cpu()
        #outputs_1 = (outputs_1 + paddle.flip(outputs_flip_1_1, [2]) + paddle.flip(outputs_flip_2_1, [3])+ paddle.flip(outputs_flip_3_1, [2,3])) / 4
        outputs_1 = outputs_1.numpy()
        
        
        
        if DEBUG:
            for i in range(targets.shape[0]):
                target = np.array(targets[i, 0]* 255, dtype = np.uint8)
                if np.max(target) < 122:
                    output = np.array(((1+outputs[i, 0])).astype(np.float32) * 255, dtype = np.uint8)
                    cv2.imwrite('output/lele_{}.jpg'.format(i), np.concatenate((target, output), axis = 0))
        #preds.append(outputs)
        preds_1.append(outputs_1)
        gts.append(targets)
    # preds = np.concatenate(preds, axis = 0)
    preds = np.concatenate(preds_1, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    # pred_min, pred_max = np.min(preds), np.max(preds)
    # preds = (preds - pred_min) / (pred_max - pred_min)
    targets = gts
    #outputs = (preds + preds_1) / 2
    outputs = preds #* preds_1
    # outputs_min, outputs_max = np.min(outputs), np.max(outputs)
    # outputs = (outputs - outputs_min) / (outputs_max - outputs_min)
    for i in range(targets.shape[0]):
        gt_list_sp.append(np.max(targets[i]))
        pr_list_sp.append(np.max(outputs[i]))
        if eval:
            outputs[i] = gaussian_filter(outputs[i], sigma=6)
        #outputs[i] = gaussian_filter(outputs[i], sigma=6)
        gt_list_px.extend(targets[i].ravel())
        pr_list_px.extend(outputs[i].ravel())
        
    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

    print("EVAL:  image-auc : {:.6f} pixel-auc:{:.6f}".format(auroc_sp, auroc_px))
    
    return auroc_px, auroc_sp


def train(args, exp_dir):
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, exp_dir, '{}'.format(args.category))
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = utils.Logger(checkpoint_dir)
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
        train_one_epoch(train_dataloader_paddle, model_paddle, optimizer_paddle, epoch, args.category, summary_writer, logger)
        #optimizer_paddle.set_lr(optimizer_paddle.get_lr() * 0.98)

        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            pixel_auroc, image_auroc = eval_once(test_dataloader_paddle, model_paddle)
            summary_writer.add_scalar("pixel_auroc/epoch", pixel_auroc, epoch)
            summary_writer.add_scalar("image_auroc/epoch", image_auroc, epoch)
            
            current_result = pixel_auroc * const.PIXEL_AUROC_RATIO + image_auroc * const.IMAGE_AUROC_RATIO
            if current_result >= max_result:
                if current_result == max_result and pixel_auroc < best_pixel_auroc:
                    count = 0
                else:
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
    logger.reset()
    return best_pixel_auroc, best_image_auroc


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model_paddle(config)
    pixel_auroc_dict, image_auroc_dict = {}, {}
    pixel_auroc_mean, image_auroc_mean = 0, 0
    for category in const.MVTEC_CATEGORIES if args.category == 'all' else [args.category]:
        args.category = category
        checkpoint_path_last = os.path.join(args.checkpoint_dir, category, 'last.pdparams')
        checkpoint_path_best = os.path.join(args.checkpoint_dir, category, 'best.pdparams')
        checkpoint = paddle.load(checkpoint_path_last if args.is_last else checkpoint_path_best)


        model.set_dict(checkpoint["model_state_dict"])
        test_dataloader = build_test_data_loader_paddle(args, config)

        pixel_auroc, image_auroc = eval_once(test_dataloader, model, eval=True)
        pixel_auroc_dict[category] = pixel_auroc
        image_auroc_dict[category] = image_auroc
        pixel_auroc_mean += pixel_auroc
        image_auroc_mean += image_auroc
    pixel_auroc_mean /= len(pixel_auroc_dict)
    image_auroc_mean /= len(image_auroc_dict)
    
    for category in pixel_auroc_dict.keys():
        print("{:15}  pixel auroc:{:.3f}   image auroc:{:.3f}".format(category, pixel_auroc_dict[category], image_auroc_dict[category]))
        
    print("{:15}  pixel auroc:{:.3f}   image auroc:{:.3f}".format('all', pixel_auroc_mean, image_auroc_mean))


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, default='configs/resnet18.yaml', help="path to config file"
    )
    parser.add_argument("--data", type=str, default='./mvtec-ad', help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES + ['all'],
        default='all',
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument("--is_last", action="store_true", help="load last param")
    parser.add_argument(
        "-ckpt_dir", "--checkpoint_dir", type=str, help="path to load checkpoint"
    )
    parser.add_argument(
        "--exp_dir",default='', type=str, help="path to save checkpoint"
    )
    
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
    
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    
    if args.exp_dir == '':
        exp_dir = "exp_{}_{}".format(const.TRAIN_BATCH_SIZE, len(os.listdir(const.CHECKPOINT_DIR)))
    else:
        exp_dir = args.exp_dir
    all_category = const.MVTEC_CATEGORIES.copy()
    pixel_auroc_list_all = {}
    image_auroc_list_all = {}
    
    if args.category == 'all':
        if args.eval:
            evaluate(args)
        else:
            for category in all_category:
                args.category = category
                
                best_pixel_auroc, best_image_auroc = train(args, exp_dir)
                pixel_auroc_list_all[category] = best_pixel_auroc
                image_auroc_list_all[category] = best_image_auroc
                print('save:{}'.format(exp_dir))
                print_result(pixel_auroc_list_all, image_auroc_list_all)
    else:
        if args.eval:
            evaluate(args)
        else:
            pixel_auroc_list, image_auroc_list = train(args, exp_dir)
            pixel_auroc_list_all[args.category] = pixel_auroc_list
            image_auroc_list_all[args.category] = image_auroc_list
            print('save:{}'.format(exp_dir))
            print_result(pixel_auroc_list_all, image_auroc_list_all)
    
