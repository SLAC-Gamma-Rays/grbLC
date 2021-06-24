from multiprocessing import Semaphore


class SynchronizedEcho:
    def __init__(self, n=1):
        self.print_lock = Semaphore(n)

    def __call__(self, *a, **b):
        self.print_lock.acquire()
        print(*a, **b)
        self.print_lock.release()
