from llava.train.train import train
import os
import builtins
import transformers
import deepspeed


def config_logger():
    import logging
    import glob
    from datetime import datetime
    rank = os.environ.get('RANK', '-1')

    def _add_file_logger(name: str, log_dir: str, level=logging.INFO):
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


def set_builtin_print(is_local_main_process: bool):
    builtins_print = builtins.print

    if 'LLAVA_DEBUG' not in os.environ:
        debug_level = 0
    else:
        debug_level = int(os.environ['LLAVA_DEBUG'])
    assert debug_level >= 0

    rank = os.environ.get('RANK', '-1')

    def custom_print(*args, rank0_only=True, **kwargs):
        if len(args) == 0:
            builtins_print(**kwargs)
            return
        now_debug_level = 0
        if isinstance(args[0], str) and args[0].startswith('[DEBUG]'):
            assert isinstance(args[1], int) and 1 <= args[1] <= 9
            now_debug_level = args[1]
        if debug_level < now_debug_level:
            return
        if not is_local_main_process and rank0_only:
            return
        builtins_print(f'[RANK{rank}]', *args, **kwargs)

    builtins.print = custom_print


if __name__ == '__main__':
    from accelerate import PartialState
    is_main_process = PartialState().is_main_process
    is_local_main_process = PartialState().is_local_main_process

    # set_builtin_print(is_local_main_process)

    config_logger()

    train(attn_implementation="flash_attention_2")
