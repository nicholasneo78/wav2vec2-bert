from transformers import PreTrainedTokenizerFast
from typing import List, Dict
from tqdm import tqdm

class BuildBPECorpus:

    '''
    To build the BPE corpus (subwords) for training the kenlm LM
    '''

    def __init__(
        self,
        input_text_file_dir: str,
        input_tokenizer_dir: str,
        output_text_file_dir: str,
    ) -> None:
        
        '''
        input_text_file_dir: the text file that combines the train and dev dataset, in text representation
        input_tokenizer_dir: the wrapped huggingface PreTrainedTokenizerFast tokenizer directory
        output_text_file_dir: the output text file in subwords representation
        '''
        
        self.input_text_file_dir = input_text_file_dir
        self.input_tokenizer_dir = input_tokenizer_dir
        self.output_text_file_dir = output_text_file_dir

    def load_raw_text_file(self) -> List[str]:

        '''
        loads the input text file that contains the annotations and store it as a list of strings (annotations)
        '''

        with open(self.input_text_file_dir, 'r+', encoding='utf-8') as fr:
            data = [entry.replace('\n', '').replace('\r', '') for entry in fr.readlines()]

        return data

    def load_tokenizer(self) -> PreTrainedTokenizerFast:

        '''
        loads the wrapped huggingface tokenizer, tokenizer file is the .json file itself, not the whole directory
        '''

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.input_tokenizer_dir,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]"
        )

        return wrapped_tokenizer
    
    def export_subwords_file(self) -> None:
        
        '''
        final text file to represent the annotations in subwords
        '''
        
        data, wrapped_tokenizer = self.load_raw_text_file(), self.load_tokenizer()

        subwords_list = []

        # split the text data into subword level
        for entry in tqdm(data):
            encoding = wrapped_tokenizer(text=entry).input_ids
            subwords = [list(wrapped_tokenizer.vocab.keys())[list(wrapped_tokenizer.vocab.values()).index(idx)] for idx in encoding]
            subwords_list.append(subwords)

        # write to subwords text file
        with open(self.output_text_file_dir, 'w+', encoding='utf-8') as fr:
            for subwords in tqdm(subwords_list):
                subword_sentence = ' '.join(subwords).replace('Ġ', '| ')
                fr.writelines(subword_sentence + '\n')

        return self.output_text_file_dir

    def __call__(self) -> None:
        return self.export_subwords_file()