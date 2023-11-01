from python.global_config import SecretConfig
import logging

class StatelessLogger:
    class __StatelessLogger:
        def __init__(self, rank):
            self.rank = rank
            self.logfile_path = SecretConfig.stateless_logfile
            logging.basicConfig(filename=self.logfile_path,
                                level=logging.DEBUG,
                                format=f'[NS][{rank}][%(asctime)s.%(msecs)03d][%(created).6f]: %(message)s',
                                datefmt='%H.%M.%S')
            self.logger = logging.getLogger(SecretConfig.stateless_logger_name)

        def debug(self, msg):
            self.logger.debug(msg)

        def info(self, msg):
            self.logger.info(msg)

        def warning(self, msg):
            self.logger.warning(msg)

        def error(self, msg):
            self.logger.error(msg)

        def critical(self, msg):
            self.logger.critical(msg)


    instance = None
    def __init__(self, rank):
        if not StatelessLogger.instance:
            StatelessLogger.instance = StatelessLogger.__StatelessLogger(rank)
        else:
            StatelessLogger.instance.rank = rank
    def __getattr__(self, name):
        return getattr(self.instance, name)
