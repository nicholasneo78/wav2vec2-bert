import json
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset, IterableDatasetDict
from .text_processing import TextPostProcessingManager

'''
List all helper functions that would be called by the main code that is related to importing or exporting of the manifest file throughout the whole repository
'''

def load_manifest_nemo(input_manifest_path: str) -> List[Dict[str, str]]:

    '''
    loads the manifest file in Nvidia NeMo format to process the entries and store them into a list of dictionaries

    the manifest file would contain entries in this format:

    {"audio_filepath": "subdir1/xxx1.wav", "duration": 3.0, "text": "shan jie is an orange cat"}
    {"audio_filepath": "subdir1/xxx2.wav", "duration": 4.0, "text": "shan jie's orange cat is chonky"}
    ---

    input_manifest_path: the manifest path that contains the information of the audio clips of interest
    ---
    returns: a list of dictionaries of the information in the input manifest file
    '''

    dict_list = []

    with open(input_manifest_path, 'rb') as f:
        for line in f:
            dict_list.append(json.loads(line))

    return dict_list

def create_huggingface_manifest(input_manifest_path: str, data_label: str) -> List[Dict[str, str]]:

    '''
    loads the list of dictionaries, preprocess the manifest format and create the finalized list of dictionaries into a json file that is ready to be accepted by the huggingface datasets class
    ---

    input_manifest_path: the manifest path that contains the information of the audio clips of interest
    ---
    returns a list of dictionaries of the information in the huggingface format
    '''    

    dict_list = load_manifest_nemo(input_manifest_path=input_manifest_path)

    data_list = []

    for entries in tqdm(dict_list):

        # creating the final data
        # 
        # dictionary that is to be saved to a json file
        data = {
            'file': f"{entries['audio_filepath']}",
                'audio': {
                    'path': f"{entries['audio_filepath']}",
                    'sampling_rate': 16000
                },
                'language': entries['language'],
                'text_raw': entries['text'],
                'text': TextPostProcessingManager(
                    label=data_label,
                    language=entries['language'],
                ).process_data(
                    text=entries['text']
            ),
                'duration': entries['duration']
        }

        data_list.append(data)

    return data_list

def export_splits(manifest_dir: str, data_list: List[Dict[str, str]]) -> None:

    '''
    outputs the respective (train, dev or test) manifest with the split data entries from a list 
    ---

    manifest_dir: the output manifest directory to be exported (train, dev or test)
    data_list: the list of splitted entries 
    '''
    
    with open(manifest_dir, 'w+', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_huggingface_manifest(train_dir: str, dev_dir: str) -> IterableDatasetDict:

    '''
    to read the train and dev json manifest file and form the transformers IterableDatasetDict for further preprocessing afterwards
    ---

    train_dir: the manifest file directory of the train set
    dev_dir: the manifest file directory of the dev set
    ---

    returns a huggingface IterableDatasetDict
    '''

    # initiate the path and form the final dataset
    data_files = {
        'train': train_dir,
        'dev': dev_dir
    }

    data = load_dataset("json", data_files=data_files, field="data", streaming=True)

    # load the manifest using vanilla json method to get the length of the train set
    with open(train_dir, 'r+', encoding='utf-8') as fr:
        json_train_data = json.load(fr)['data']

    return data, len(json_train_data)


def load_huggingface_manifest_evaluation(test_dir: str) -> IterableDatasetDict:
        
    '''
    to read the test json manifest file and form the transformers IterableDatasetDict for further preprocessing afterwards
    ---

    test_dir: the manifest file directory of the test set
    ---

    returns a huggingface IterableDatasetDict
    '''

    # initiate the path and form the final dataset
    data_files = {
        'test': test_dir,
    }

    data = load_dataset("json", data_files=data_files, field="data", streaming=True)

    return data

