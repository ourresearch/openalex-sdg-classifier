import boto3
import tempfile
from nltk import tokenize
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import convert_to_tensor
from transformers import TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Load the ML model from s3
s3 = boto3.client('s3')
fp = tempfile.NamedTemporaryFile()
s3.download_file('aurora-sdg-classifier', 'aurora_multilabel_v5.h5', fp.name)
model = load_model(fp.name, custom_objects={'TFBertMainLayer': TFBertModel})
fp.close()

goal_names = {
    "Goal 1": "No poverty",
    "Goal 2": "Zero hunger",
    "Goal 3": "Good health and well-being",
    "Goal 4": "Quality Education",
    "Goal 5": "Gender equality",
    "Goal 6": "Clean water and sanitation",
    "Goal 7": "Affordable and clean energy",
    "Goal 8": "Decent work and economic growth",
    "Goal 9": "Industry, innovation and infrastructure",
    "Goal 10": "Reduced inequalities",
    "Goal 11": "Sustainable cities and communities",
    "Goal 12": "Responsible consumption and production",
    "Goal 13": "Climate action",
    "Goal 14": "Life below water",
    "Goal 15": "Life in Land",
    "Goal 16": "Peace, Justice and strong institutions",
    "Goal 17": "Partnerships for the goals"
}


def tokenize_abstracts(abstracts):
    """For a given texts, adds '[CLS]' and '[SEP]' tokens
    at the beginning and the end of each sentence, respectively.
    """
    t_abstracts = []
    for abstract in abstracts:
        t_abstract = "[CLS] "
        for sentence in tokenize.sent_tokenize(abstract):
            t_abstract = t_abstract + sentence + " [SEP] "
        t_abstracts.append(t_abstract)
    return t_abstracts


def b_tokenize_abstracts(t_abstracts, max_len=512):
    """Tokenizes sentences with the help
    of a 'bert-base-multilingual-uncased' tokenizer.
    """
    b_t_abstracts = [tokenizer.tokenize(_)[:max_len] for _ in t_abstracts]
    return b_t_abstracts


def convert_to_ids(b_t_abstracts):
    """Converts tokens to its specific
    IDs in a bert vocabulary.
    """
    input_ids = [tokenizer.convert_tokens_to_ids(_) for _ in b_t_abstracts]
    return input_ids


def tokenize_abstract(abstract):
    """For a given texts, adds '[CLS]' and '[SEP]' tokens
    at the beginning and the end of each sentence, respectively.
    """
    t_abstracts = []
    t_abstract = "[CLS] "
    for sentence in tokenize.sent_tokenize(abstract):
        t_abstract = t_abstract + sentence + " [SEP] "
    t_abstracts.append(t_abstract)
    return t_abstract


def abstracts_to_ids(abstracts):
    """Tokenizes abstracts and converts
    tokens to their specific IDs
    in a bert vocabulary.
    """
    tokenized_abstracts = tokenize_abstracts(abstracts)
    b_tokenized_abstracts = b_tokenize_abstracts(tokenized_abstracts)
    ids = convert_to_ids(b_tokenized_abstracts)
    return ids


def pad_ids(input_ids, max_len=512):
    """Padds sequences of a given IDs.
    """
    p_input_ids = pad_sequences(input_ids,
                                maxlen=max_len,
                                dtype="long",
                                truncating="post",
                                padding="post")
    return p_input_ids


def create_attention_masks(inputs):
    """Creates attention masks
    for a given sequences.
    """
    masks = []
    for seq in inputs:
        seq_mask = [i > 0 for i in seq]
        masks.append(seq_mask)
    return tf.cast(masks, tf.int32)


def prepare_input(abstracts):
    ids = abstracts_to_ids(abstracts)
    padded_ids = pad_ids(ids)
    masks = create_attention_masks(padded_ids)
    return convert_to_tensor(padded_ids), convert_to_tensor(masks)


def get_predictions(abstract):
    # Make a prediction
    # Assuming 'abstracts' is a list of abstracts
    abstracts = [abstract]
    input_ids, masks = prepare_input(abstracts=abstracts)

    # Assuming your model takes two inputs
    predictions = model.predict([input_ids, masks])
    response = []
    for index, sdg_value in enumerate(predictions[0]):
        sdg_number = index + 1
        sdg_id = 'http://metadata.un.org/sdg/' + str(sdg_number)
        sdg_label = 'Goal ' + str(sdg_number)
        sdg_name = goal_names[sdg_label]
        response.append(
            {"prediction": float(sdg_value),
             "sdg": {
                 "@type": "sdg",
                 "id": sdg_id,
                 "label": sdg_label,
                 "code": str(sdg_number),
                 "name": sdg_name,
                 "type": "Goal",
             }})
        # order by prediction
        response = sorted(response, key=lambda k: k['prediction'], reverse=True)
    return response

