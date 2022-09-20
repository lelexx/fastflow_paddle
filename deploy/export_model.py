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


import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle
import configs.constants as const
from datasets.dataset import MVTecDataset
from models.fastflow import FastFlow
from utils.utils import *
import argparse
import yaml

paddle.set_device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, default='configs/resnet18.yaml', help="path to config file"
    )

    parser.add_argument(
        "--exp_dir", type=str, default = 'exp', help="path to load checkpoint"
    )
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        default='bottle',
        help="category name in mvtec",
    )
    parser.add_argument(
        '--save_inference_dir', type = str, default='deploy/inference', help='path where to save')
    

    args = parser.parse_args()
    return args


def build_model_paddle(config):
    '''
    create fastflow model
    '''
    model = FastFlow(
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
def export(args):
    os.makedirs(args.save_inference_dir, exist_ok=True)
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model_paddle(config)
    checkpoint_path = os.path.join(const.CHECKPOINT_DIR, args.exp_dir, args.category, 'best.pdparams')
    checkpoint = paddle.load(checkpoint_path)
    
    ### load pretrained checkpoint
    model.set_dict(checkpoint["model_state_dict"])
    model.eval()
    
    shape = [1, 3, config["input_size"], config["input_size"]]
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print('inference model has been saved into {}'.format(args.save_inference_dir))



if __name__ == "__main__":
    args = parse_args()
    export(args)