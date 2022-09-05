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

from pathlib import Path
import argparse
import pickle
import paddle
import yaml
from paddle import inference
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2
from paddle.vision import transforms
import fastflow
import constants as const


class InferenceEngine(object):
    """InferenceEngine

    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args
        self.model_yaml = yaml.safe_load(open(args.config, "r"))

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor\
            = self.load_predictor(
            os.path.join(args.save_inference_dir, "inference.pdmodel"),
            os.path.join(args.save_inference_dir, "inference.pdiparams"))

        # build transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.model_yaml["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.model_yaml["input_size"],
                                   self.model_yaml["input_size"]).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])


        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path):
        img = Image.open(img_path)
        img = np.array(img)
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, x):
        output = x[0]
        return output

    def run(self, x):

        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, default='configs/resnet18.yaml', help="path to config file"
    )

    parser.add_argument("--image_path", type=str, default='images/bottle_bad.png', help="path to image")
    parser.add_argument(
        "--benchmark", default=False, type=str2bool,  help="benchmark")

    parser.add_argument("--image_threshold", type=float, default= 0.8, help="image_level threshold")
    parser.add_argument("--pixel_threshold", type=float, default= 0.8, help="pixel_level threshold")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")
    
    parser.add_argument(
        "--save_inference_dir", default="deploy/inference", help="inference model dir")
    parser.add_argument(
        "--use_gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--max_batch_size", default=16, type=int, help="max_batch_size")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")



    
    args = parser.parse_args()
    return args


def infer_main(args):
    global autolog
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="fastflow",
            inference_config=inference_engine.config,
            gpu_ids="auto" if 'gpu' in const.device else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = inference_engine.preprocess(args.image_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    output = inference_engine.postprocess(output)
    
    image_score = np.max(output)
    if image_score > args.image_threshold:
        print('异常 - score:  {:.3f}'.format(image_score))
    else:
        print('正常 - score:  {:.3f}'.format(image_score))
    output = gaussian_filter(output, sigma=6)[0]
    predict_map = (output > args.pixel_threshold).astype(np.float32)
    save_image = np.concatenate((output, predict_map), axis = 0) * 255

    cv2.imwrite('./output/lele.jpg', save_image.astype(np.uint8))

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()



if __name__ == "__main__":
    args = parse_args()
    infer_main(args)
