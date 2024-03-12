from typing import Dict, List
from tqdm import tqdm
import numpy as np
import logging

from jiwer import wer, cer, mer
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class WER:

    '''
    WER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return wer(reference=self.references, hypothesis=self.predictions)

class CER:
    
    '''
    CER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return cer(reference=self.references, hypothesis=self.predictions)

class MER:
    
    '''
    MER metrics
    '''

    def __init__(self, predictions=None, references=None):
        self.predictions = predictions
        self.references = references

    
    def compute(self):
        return mer(reference=self.references, hypothesis=self.predictions)
    
def compute_metrics(pred, tokenizer: PreTrainedTokenizerFast) -> Dict[str, float]:

    '''
    to evaluate the wer of the model on the validation set during finetuning
    ---

    pred: predicted transcription from the validation set
    tokenizer: the processor object to do tokenization of the dataset
    ---
    returns a dictionary with the value as the WER value
    '''

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # we do not want to group tokens when computing the metrics
    label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False, skip_special_tokens=True)

    logging.getLogger('INFO').info(f'Pred str raw: {pred_str[:5]}\n')
    logging.getLogger('INFO').info(f'Label str raw: {label_str[:5]}\n')

    get_wer = WER(predictions=pred_str, references=label_str)
    get_cer = CER(predictions=pred_str, references=label_str)
    get_mer = MER(predictions=pred_str, references=label_str)

    return {
        "wer": get_wer.compute(), 
        "cer": get_cer.compute(),
        "word_acc": 1-get_mer.compute(),
    }