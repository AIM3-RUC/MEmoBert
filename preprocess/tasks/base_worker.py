import abc
import sys

class BaseWorker(object, metaclass=abc.ABCMeta):
    def __init__(self, logger=None):
        self.logger = logger

    def print(self, *obj):
        class_name = self.__class__.__name__
        print_str = ' '.join([x.__str__() for x in obj])
        if self.logger:
            self.logger.info('[{}]: {}'.format(class_name, print_str))
        else:
            print('[{}]: {}'.format(class_name, print_str))