import logging
import os


class SaveHandle:
    """handle the number of"""

    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        assert len(self.save_list) <= self.max_num

        if len(self.save_list) == self.max_num:
            remove_path, self.save_list = self.save_list[0], self.save_list[1:]
            if os.path.exists(remove_path):
                os.remove(remove_path)

        self.save_list.append(save_path)


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
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


class Logger:
    def __init__(self, log_file):
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def print_config(self, config):
        """Print configuration of the model"""
        for k, v in config.items():
            self.logger.info(f"{k.ljust(15)}:\t{v}")

    def info(self, msg):
        self.logger.info(msg)
