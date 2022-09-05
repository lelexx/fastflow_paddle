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
    save_dir = os.path.join(checkpoint_dir, 'files')
    os.makedirs(save_dir, exist_ok=True)
    os.system('cp -r ./configs {}'.format(save_dir))
    os.system('cp ./*.py {}'.format(save_dir))

    
if __name__ == '__main__':
    logger = Logger('.')
    print('lele')
    logger.print('lele')
    print('xxxxx')
    print('yyyyyy')
    logger.print('lele')
    logger.print(['xx', 'tttt'])
    logger.reset()
