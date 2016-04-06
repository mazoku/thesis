import time
import progressbar

bar = progressbar.ProgressBar()#redirect_stdout=True)
for i in range(100):
    # print 'Some text', i
    time.sleep(0.1)
    bar.update(i)