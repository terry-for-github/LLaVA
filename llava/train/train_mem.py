from llava.train.train import train
import os


def config_logger():
    import logging
    import glob
    from datetime import datetime
    rank = os.environ.get('RANK', '-1')

    def _add_file_logger(name: str, log_dir: str, level=logging.DEBUG):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.propagate = False
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                      "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
        log_filename = os.path.join(log_dir, f'{logger.name}_{rank:0>2}.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = f"logs/log_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    _add_file_logger('transformers', log_dir)
    _add_file_logger('DeepSpeed', log_dir)

    if rank in ['0', '-1']:
        import shutil
        log_files = sorted(glob.glob(os.path.join('logs', "log_*")), key=os.path.getctime)

        while len(log_files) > 20:
            oldest_log = log_files.pop(0)
            shutil.rmtree(oldest_log)


if __name__ == "__main__":
    # config_logger()
    train(attn_implementation="flash_attention_2")
