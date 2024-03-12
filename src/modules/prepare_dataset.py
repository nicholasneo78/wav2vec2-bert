import librosa
from transformers import PreTrainedTokenizerFast, SeamlessM4TFeatureExtractor
import os

def prepare_dataset(
        batch, 
        tokenizer: PreTrainedTokenizerFast,
        feature_extractor: SeamlessM4TFeatureExtractor,
        root_path: str,
    ):
        
    '''
    to prepare the final dataset that is to be fed into the pretrained whisper model for finetuning, this method is used in the huggingface dataset.map(...) call
    ---

    batch: the batch of data that would be processed
    tokenizer: tokenize the text

    root_path: the absolute path to be added to join with the relative paths from the manifest files
    ---
    
    returns the batch of data after being processed
    '''

    # some filepath preprocessing
    batch['relative_file_path'] = batch['file']
    batch['file'] = os.path.join(root_path, batch['file'])

    # retrieve the audio features from the filepath
    audio = batch["audio"]
    audio['path'] = os.path.join(root_path, audio['path'])
    audio["array"] = librosa.load(audio["path"], sr=audio["sampling_rate"])[0]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["input_length"] = len(batch["input_features"])
    batch["labels"] = tokenizer(text=batch["text"]).input_ids
    
    return batch