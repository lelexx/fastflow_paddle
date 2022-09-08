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


import os, sys

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    '''
    save print results to train.log
    '''
    def __init__(self, checkpoint_dir, filename="train.log"):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(os.path.join(checkpoint_dir, filename), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def print(self, messages):
        if isinstance(messages, str):
           messages = [messages] 
        for message in messages:
            self.log.write(message + '\r\n')
        
    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

def save_config(checkpoint_dir):
    '''
    save python files and yamls in checkpoint
    '''
    save_dir = os.path.join(checkpoint_dir, 'files')
    os.makedirs(save_dir, exist_ok=True)
    os.system('cp -r ./configs {}'.format(save_dir))
    os.system('cp ./*.py {}'.format(save_dir))

    

