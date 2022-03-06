import logging
import os

class logger:
    def setlogger(self, log_filename):
        if len(logging.getLogger().handlers) == 0:
            log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
            # 打印到控制台
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)  # 设置控制台日志输出的级别。如果设置为logging.INFO，就不会输出DEBUG日志信息
            console.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(console)

        if len(logging.getLogger().handlers) > 1:
            logger = logging.getLogger()
            logger.handlers[1].stream.close()
            logger.removeHandler(logger.handlers[1])

        log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
        logging.getLogger().setLevel(logging.DEBUG)

        # 自动换文件
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    @staticmethod
    def set_logging(LOG_FILE, clean=False):
        if clean:
            if os.path.isfile(LOG_FILE):
                with open(LOG_FILE, 'w') as f:
                    pass

        logging = logging.getlogging()
        logging.basicConfig(filename=LOG_FILE,
                            format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')

        logging.setLevel(logging.DEBUG)

        # ch = logging.StreamHandler()
        # logging.addHandler(ch)

        return logging