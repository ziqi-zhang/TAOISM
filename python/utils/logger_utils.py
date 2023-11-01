# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

import sys


class Logger(object):
    logfile_path = "logfile.log"

    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(self.logfile_path, "a")

    def reset_logfile(self, path):
        self.logfile_path = path
        self.log = open(self.logfile_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        # pass
        self.terminal.flush()
        self.log.flush()

# sys.stdout = Logger()
