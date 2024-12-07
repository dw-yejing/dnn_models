import loguru 
import sys
from typing import Literal
from pathlib import Path

class LoguruLogger:
    def __init__(self, log_folder:str="log", level:Literal["DEBUG", "INFO", "WARNING", "ERROR"]="INFO", format:str="{time} {level} {message}"):
        self.logger = loguru.logger
        self.info_file = f"{log_folder}/info.log"
        self.warning_file = f"{log_folder}/warning.log"
        self.error_file = f"{log_folder}/error.log"
        self.debug_file = f"{log_folder}/debug.log"
        
        self.level = level
        self.levels = {
            "TRACE": 5,
            "DEBUG": 10,
            "INFO": 20,
            "SUCCESS": 25,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        self.format = format
        if Path(log_folder).exists() is False:
            Path(log_folder).mkdir(parents=True, exist_ok=True) 
        self._configure_logger()
        
    def _configure_logger(self):
        self.logger.remove()
        self.logger.add(sys.stdout, level=self.levels["INFO"], format=self.format)
        self.logger.add(sys.stdout, level=self.levels["ERROR"], format=self.format)
        self.logger.add(self.info_file, level=self.levels["INFO"], format=self.format, rotation="1 week") 
        self.logger.add(self.error_file, level=self.levels["ERROR"], format=self.format, rotation="1 week") 
        
    def add(self, 
            log_path:str="log.log", 
            level:Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]="INFO",
            format:str="{time} {level} {message}", 
            rotation:Literal["50 MB", "100 MB", "200 MB",]="100 MB",
            **kwargs):
        self.logger.add(log_path, level=level, format=format, rotation=rotation, **kwargs)
        
    def remove(self, log_path:str):
        self.logger.remove(log_path)

loguru_logger = LoguruLogger()
logger = loguru_logger.logger
