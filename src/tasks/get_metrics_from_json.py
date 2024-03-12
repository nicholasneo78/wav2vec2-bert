import os
from typing import List, Dict
from modules import WER, CER, MER, load_manifest_nemo
import logging

# Setup logging in a nice readable format
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class GetMetricsFromJSON:

    '''
    to get the WER from the JSON file with key "prediction" and "ground truth" after running the evaluate_model.py script to generate the json file, can use this script to filter test subset
    '''

    def __init__(self, cfg: Dict, is_remote: bool) -> None:

        '''
        input_json_path (str): the json directory that was generated from evaluate_model.py
        '''

        args = cfg.get_metrics_from_json

        if not is_remote:

            self.config_dict = {
                'input_manifest_path': args.input_json_path,
                'type': args.filter.type,
                'query': args.filter.query,
            }

        else:

            self.config_dict = {
                'input_manifest_path': os.path.join(args.temp.manifest_path, args.manifest.manifest_path),
                'type': args.filter.type,
                'query': args.filter.query,
            }

    def get_metrics(self) -> None:

        '''
        main method to output the WER, CER and Word Accuracy
        '''

        data = load_manifest_nemo(input_manifest_path=self.config_dict['input_manifest_path'])

        if not self.config_dict['type']:
            pred_list = [pred['pred_str'] for pred in data]
            pred_lm_list = [pred['pred_str_with_lm'] for pred in data]
            ground_truth_list = [ref['text'] for ref in data]
        else:
            pred_list = []
            pred_lm_list = []
            ground_truth_list = []
            if self.config_dict['type'] == 'language':
                for lang in self.config_dict['query']:
                    preds = [pred['pred_str'] for pred in data if lang in pred['language']]
                    preds_lm = [pred['pred_str_with_lm'] for pred in data if lang in pred['language']]
                    grounds = [ref['text'] for ref in data if lang in ref['language']]
                    pred_list.extend(preds)
                    pred_lm_list.extend(preds_lm)
                    ground_truth_list.extend(grounds)
            elif self.config_dict['type'] == 'data':
                for data_name in self.config_dict['query']:
                    preds = [pred['pred_str'] for pred in data if data_name in pred['audio_filepath']]
                    preds_lm = [pred['pred_str_with_lm'] for pred in data if data_name in pred['audio_filepath']]
                    grounds = [ref['text'] for ref in data if data_name in ref['audio_filepath']]
                    pred_list.extend(preds)
                    pred_lm_list.extend(preds_lm)
                    ground_truth_list.extend(grounds)
        
        # compute the WER
        get_wer = WER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_cer = CER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_mer = MER(
            predictions=pred_list,
            references=ground_truth_list
        )

        get_wer_lm = WER(
            predictions=pred_lm_list,
            references=ground_truth_list
        )

        get_cer_lm = CER(
            predictions=pred_lm_list,
            references=ground_truth_list
        )

        get_mer_lm = MER(
            predictions=pred_lm_list,
            references=ground_truth_list
        )

        print()
        logging.getLogger('INFO').info("NO LM")
        logging.getLogger('INFO').info("Test WER: {:.5f}".format(get_wer.compute()))
        logging.getLogger('INFO').info("Test CER: {:.5f}".format(get_cer.compute()))
        logging.getLogger('INFO').info("Test Word Accuracy: {:.5f}\n".format(1-get_mer.compute()))

        logging.getLogger('INFO').info("WITH LM")
        logging.getLogger('INFO').info("Test WER: {:.5f}".format(get_wer_lm.compute()))
        logging.getLogger('INFO').info("Test CER: {:.5f}".format(get_cer_lm.compute()))
        logging.getLogger('INFO').info("Test Word Accuracy: {:.5f}\n".format(1-get_mer_lm.compute()))

    def __call__(self):
        return self.get_metrics()