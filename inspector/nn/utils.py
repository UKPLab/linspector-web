from allennlp.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser
from allennlp.models.crf_tagger import CrfTagger
from allennlp.models.esim import ESIM
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.predictors import BiaffineDependencyParserPredictor, SentenceTaggerPredictor

from .dataset_readers.intrinsic_dataset_reader import IntrinsicDatasetReader
from .predictors.esim_predictor import ESIMPredictor

from enum import Enum, unique

@unique
class Classifier(Enum):
    """Enum containing supported AllenNLP classifiers."""
    BIAFFINE_PARSER = 'BiaffineDependencyParser'
    CRF_TAGGER = 'CrfTagger'
    ESIM = 'ESIM'
    SIMPLE_TAGGER = 'SimpleTagger'

    def __str__(self):
        return self.value

def get_predictor_for_model(model, field_key):
    """Get a matching predictor for a model.

    Args:
        model: An allenlp.models.model.
        field_key: Set the instance key returned from the dataset reader.

    Returns:
        An allennlp.predictors.predictor for the model.

    Raises:
        NotImplementedError: The model type is not supported.
    """
    if isinstance(model, BiaffineDependencyParser):
        return BiaffineDependencyParserPredictor(model, UniversalDependenciesDatasetReader())
    elif isinstance(model, CrfTagger) or isinstance(model, SimpleTagger):
        return SentenceTaggerPredictor(model, IntrinsicDatasetReader(field_key=field_key))
    elif isinstance(model, ESIM):
        return ESIMPredictor(model, IntrinsicDatasetReader(field_key=field_key))
    else:
        raise NotImplementedError
