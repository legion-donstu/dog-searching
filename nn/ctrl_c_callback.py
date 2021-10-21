from tensorflow import keras
import signal

is_sigint = False


class CtrlCStopping(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if is_sigint:
            self.model.stop_training = True


def handler(signum, frame):
    print()
    print("Training will be stopped at the end of the epoch")
    global is_sigint
    is_sigint = True


signal.signal(signal.SIGINT, handler)
