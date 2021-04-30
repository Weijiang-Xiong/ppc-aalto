import sys


class Logger:
    def __init__(self):
        self.level = 0

    def log_command(self, args, level=1):
        if self.level >= level:
            msg = '>> ' + ' '.join(args)
            if sys.stdout.isatty():
                print('\033[34m' + msg + '\033[0m', flush=True)
            else:
                print(msg, flush=True)


logger = Logger()


def increase_log_level():
    logger.level += 1


def set_log_level(level):
    logger.level = level


def log_command(args, level=1):
    logger.log_command(args, level)
