from __future__ import print_function
from time import time
from enum import IntEnum

class VerboseLevel(IntEnum):
    EVERY = 1
    LAYER = 2
    RUN = 3
    EPOCH = 4

def show_time_diff(name, t0, t1):
    print(name+":", f"{(t1 - t0) * (10 ** 3):.4f}", "ms")


class Timer(object):
    def __init__(self, name, verbose_level=VerboseLevel.EVERY):
        self.name = name
        self.t0 = time()
        self.t1 = None
        self.verbose_level = verbose_level

    def start(self):
        self.t0 = time()

    def show_time(self, tmp_name=None):
        showName = self.name if tmp_name is None else tmp_name
        show_time_diff("Time for " + showName, self.t0, self.t1)

    def end(self, tmp_name=None, is_show_time=None):
        if is_show_time is None:
            threshold = NamedTimer.get_instance().verbose_level
            timer_level = self.verbose_level
            is_show_time = timer_level >= threshold
        self.t1 = time()
        if is_show_time:
            send_name = self.name + " - " + tmp_name if tmp_name is not None else None
            self.show_time(tmp_name=send_name)

    def get_elapse_time(self):
        elapse_time = (self.t1 - self.t0) * (10 ** 3)
        return elapse_time


class NamedTimer(object):
    __instance = None

    @staticmethod
    def get_instance():
        if NamedTimer.__instance is None:
            NamedTimer()
        return NamedTimer.__instance

    def __init__(self):
        NamedTimer.__instance = self
        self.timers = {}
        self.verbose_level = VerboseLevel.EVERY

    @staticmethod
    def start_timer(name, **kwargs):
        NamedTimer.get_instance().timers[name] = Timer(name, **kwargs)
        return NamedTimer.get_instance().timers[name]

    @staticmethod
    def start(name, **kwargs):
        return NamedTimer.get_instance().start_timer(name, **kwargs)

    @staticmethod
    def end_timer(name, **kwargs):
        NamedTimer.get_instance().timers[name].end(**kwargs)

    @staticmethod
    def end(name, tmp_name=None):
        # print(NamedTimer.get_instance().timers[name].verbose_level, NamedTimer.get_instance().verbose_level)
        NamedTimer.get_instance().end_timer(name, tmp_name=tmp_name)

    @staticmethod
    def set_verbose_level(verbose_level):
        if not isinstance(verbose_level, VerboseLevel):
            raise ValueError("Please set an enum from VerboseLevel")
        NamedTimer.get_instance().verbose_level = verbose_level


class NamedTimerInstance(object):
    def __init__(self, name, verbose_level=VerboseLevel.EVERY):
        self.name = name
        self.verbose_level = verbose_level

    def __enter__(self):
        return NamedTimer.start(self.name, verbose_level=self.verbose_level)
        ...

    def __exit__(self, *args):
        NamedTimer.end(self.name)
        ...

class NetworkNamedTimerInstance(object):
    def __init__(self, name, verbose_level=VerboseLevel.EVERY):
        self.name = name
        self.verbose_level = verbose_level

    def __enter__(self):
        return NamedTimer.start(self.name, verbose_level=self.verbose_level)

    def __exit__(self, *args):
        NamedTimer.end(self.name)
