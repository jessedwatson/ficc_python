import tensorflow as tf

def cosine_annealing(min, max, half_cycle_epochs):
    def scheduler(epoch, lr):
        t = (tf.cos(float(epoch) * 3.1415926 / half_cycle_epochs) * 0.5 + 0.5)
        return t * min + (1.0 - t) * max
    return scheduler

def warmup_lr(start, target, n_epochs):
    def scheduler(epoch, lr):
        if epoch <= n_epochs:
            return float(epoch * (target - start)) / n_epochs + start
        else:
            return lr
    return scheduler

