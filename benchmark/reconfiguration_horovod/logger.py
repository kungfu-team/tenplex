import time
import numpy as np


class Logger(object):

    def __init__(self, log_period=10) -> None:
        self.t0 = time.time()
        self.img_secs = []
        self.step = 0
        self.trained = 0
        self.log_period = log_period

    def add(self, trained):
        self.step += 1
        self.trained += trained

        if self.step % self.log_period == 0:
            t1 = time.time()
            took = t1 - self.t0

            img_sec = self.trained / took
            self.t0 = t1
            self.trained = 0

            print('step #%d : %.1f img/sec' % (self.step, img_sec))
            self.img_secs.append(img_sec)

    def report(self):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per: %.1f +-%.1f' % (img_sec_mean, img_sec_conf))
