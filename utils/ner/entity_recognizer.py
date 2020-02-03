import anago
from anago.utils import load_data_and_labels, load_glove

MODEL_DIRPATH = "../utils/ner/model/"
WEIGHT_FILENAME = "nerweight.h5"
PARAMS_FILENAME = "nerparams.json"
PREPROCESSOR_FILENAME = "nerprepro.pkl"

model = anago.Sequence().load(weights_file=MODEL_DIRPATH + WEIGHT_FILENAME, 
            params_file=MODEL_DIRPATH + PARAMS_FILENAME, 
            preprocessor_file=MODEL_DIRPATH + PREPROCESSOR_FILENAME)


def __get_ner_tags(result):
    ner_tags = ["O" for _ in result['words']]
    for entity in result['entities']:
        ner_tags[entity['beginOffset']] = entity['type']
    return ner_tags
def get_entities(sentence):
    result = model.analyze(sentence)
    return __get_ner_tags(result)