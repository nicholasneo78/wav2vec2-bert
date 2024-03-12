import json
from typing import List

class BuildSentences:

    """
    To output the train and dev combined manifest files in .txt format to build the BPE tokenizer and the kenlm LM 
    """

    def __init__(
        self, 
        output_text_dir: str, 
        train_manifest_dir: str, 
        dev_manifest_dir: str
    ):
        
        self.output_text_dir = output_text_dir
        self.train_manifest_dir = train_manifest_dir
        self.dev_manifest_dir = dev_manifest_dir

    def load_manifest_for_bpe(self, input_manifest_dir: str) -> List[str]:

        '''
        load the huggingface format dataset manifest and get ready to be called to export the text file
        ---
        input_manifest_dir: directory to huggingface manifest file

        returns a list of annotations
        '''

        with open(input_manifest_dir, 'r+') as fr:
            data = json.load(fr)
        output_text_list = [entry['text'] for entry in data['data']]

        return output_text_list

    def export_text_file(self) -> None:
        
        '''
        takes in the train and dev manifest file in huggingface format, then output the transcriptions in a text file
        ---
        output_text_dir: output filepath of the text file
        train_manifest_dir: input train manifest filepath
        dev_manifest_dir: input dev manifest filepath 
        '''

        train_list = self.load_manifest_for_bpe(input_manifest_dir=self.train_manifest_dir)
        dev_list = self.load_manifest_for_bpe(input_manifest_dir=self.dev_manifest_dir)
        
        combined_list = train_list + dev_list

        with open(self.output_text_dir, 'w+') as fw:
            for idx, text in enumerate(combined_list):
                fw.write(text+'\n') 

        return self.output_text_dir

    def __call__(self) -> None:
        return self.export_text_file()