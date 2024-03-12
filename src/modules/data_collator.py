from typing import List, Union, Dict
from dataclasses import dataclass
import torch

from transformers import PreTrainedTokenizerFast, SeamlessM4TFeatureExtractor

@dataclass
class DataCollatorCTCWithPadding:
    
    feature_extractor: SeamlessM4TFeatureExtractor
    tokenizer: PreTrainedTokenizerFast
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        audio_filepath = [{"audio_filepath": feature["relative_file_path"]} for feature in features]
        language = [{"language": feature["language"]} for feature in features] 
        duration = [{"duration": round(float(feature["duration"]), 5)} for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.tokenizer.pad(
            encoded_inputs=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["audio_filepath"] = [entry['audio_filepath'] for entry in audio_filepath]
        batch['language'] = [entry['language'] for entry in language]
        batch['duration'] = [entry['duration'] for entry in duration]

        return batch