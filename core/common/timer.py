import time


class Timer:
    def __init__(self, print_message=True):
        self.print_message = print_message

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        if self.print_message:
            print('timed block took %.03f sec.' % self.interval)
