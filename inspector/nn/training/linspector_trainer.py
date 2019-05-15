from allennlp.training.trainer import Trainer

class LinspectorTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = []

    def _train_epoch(self, epoch):
        for callback in self._callbacks:
            callback(1 / self._num_epochs * epoch)
        return super()._train_epoch(epoch)

    def subscribe(self, callback):
        """Subscribe with callback to get epoch progress [0, 1]."""
        self._callbacks.append(callback)
