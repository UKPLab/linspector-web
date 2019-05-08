from allennlp.modules import FeedForward
from allennlp.nn.activations import Activation

from .linspector_linear import LinspectorLinear

class MultilayerPerceptron(LinspectorLinear):

    def __init__(self, word_embeddings, vocab, num_layers = 2):
        super().__init__(word_embeddings, vocab)
        self.hidden2tag = FeedForward(input_dim=self.word_embeddings.get_output_dim(), num_layers=num_layers, hidden_dims=[self.word_embeddings.get_output_dim(), self.num_classes], activations=[Activation.by_name('relu')(), Activation.by_name('linear')()], dropout=[0.5, 0.0])
