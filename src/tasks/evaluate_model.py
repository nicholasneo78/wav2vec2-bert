from typing import Dict
from tqdm import tqdm
import numpy as np
import logging
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, PreTrainedTokenizerFast, set_seed

from modules import load_huggingface_manifest_evaluation, prepare_dataset, DataCollatorCTCWithPadding, WER, CER, MER, TextPostProcessingManager
from pyctcdecode_ import build_ctcdecoder

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class EvaluateModel:

    """
    Loads the test dataset manifest Huggingface format and starts the evaluation
    """

    def __init__(self, cfg: Dict, is_remote: bool) -> None:
        
        args = cfg.evaluate_model

        if not is_remote:

            self.config_dict = {
                'data': {
                    'test_root_path': args.data.root_path,
                    'test_manifest_dir': args.data.test_manifest_dir,
                    'data_label': args.data.data_label
                },
                'model': {
                    'seed': args.model.seed,
                    'finetuned_model_path': args.model.saved_model_path,
                    'use_safetensors': args.model.use_safetensors,
                    'attention_dropout': args.model.attention_dropout,
                    'hidden_dropout': args.model.hidden_dropout,
                    'feat_proj_dropout': args.model.feat_proj_dropout,
                    'final_dropout': args.model.final_dropout,
                    'layerdrop': args.model.layerdrop,
                    'feature_extractor_path': args.model.feature_extractor_path,
                    'tokenizer_path': args.model.tokenizer_path,
                    'data_loader_batch_size': args.model.data_loader_batch_size,
                    'activate_dropout': args.model.activate_dropout
                },
                'results': {
                    'output_pred_path': args.results.output_pred_path
                },
                'kenlm': {
                    'model_path': args.kenlm.model_path,
                    'alpha': args.kenlm.alpha,
                    'beta': args.kenlm.beta,
                    'beam_width': args.kenlm.beam_width,
                    'hotwords': args.kenlm.hotwords,
                    'hotword_weight': args.kenlm.hotword_weight,
                },
            }
        
        else:

            self.config_dict = {
                'data': {
                    'test_root_path': args.temp.dataset_test_path,
                    'test_manifest_dir': os.path.join(args.temp.dataset_test_path, args.data.test.input_manifest_path),
                    'data_label': args.data.data_label
                },
                'model': {
                    'seed': args.model.seed,
                    'finetuned_model_path': os.path.join(args.temp.model_finetuned_path, args.model.model_path),
                    'use_safetensors': args.model.use_safetensors,
                    'attention_dropout': args.model.attention_dropout,
                    'hidden_dropout': args.model.hidden_dropout,
                    'feat_proj_dropout': args.model.feat_proj_dropout,
                    'final_dropout': args.model.final_dropout,
                    'layerdrop': args.model.layerdrop,
                    'feature_extractor_path': os.path.join(args.temp.model_finetuned_path, args.model.feature_extractor_path),
                    'tokenizer_path': os.path.join(args.temp.tokenizer_path, args.tokenizer.tokenizer_path),
                    'data_loader_batch_size': args.model.data_loader_batch_size,
                    'activate_dropout': args.model.activate_dropout
                },
                'results': {
                    'output_pred_path': args.results.output_pred_path
                },
                'kenlm': {
                    'model_path': os.path.join(args.temp.lm_path, args.kenlm.lm_path),
                    'alpha': args.kenlm.alpha,
                    'beta': args.kenlm.beta,
                    'beam_width': args.kenlm.beam_width,
                    'hotwords': args.kenlm.hotwords,
                    'hotword_weight': args.kenlm.hotword_weight,
                },
            }


    def start_evaluation(self) -> None:

        """
        main method to load the manifest file, grabs the test data and send the data for finetuning
        """

        set_seed(self.config_dict['model']['seed'])

        dataset = load_huggingface_manifest_evaluation(
            test_dir=self.config_dict['data']['test_manifest_dir']
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Wav2Vec2BertForCTC.from_pretrained(
            self.config_dict['model']['finetuned_model_path'],
            use_safetensors=self.config_dict['model']['use_safetensors'],
            attention_dropout=self.config_dict['model']['attention_dropout'],
            hidden_dropout=self.config_dict['model']['hidden_dropout'],
            feat_proj_dropout=self.config_dict['model']['feat_proj_dropout'],
            final_dropout=self.config_dict['model']['final_dropout'],
            layerdrop=self.config_dict['model']['layerdrop'],
        ).to(device)

        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            self.config_dict['model']['feature_extractor_path']
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.config_dict['model']['tokenizer_path'],
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
        )

        data_collator = DataCollatorCTCWithPadding(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer,
            padding=True
        )

        dataset = dataset.map(
            lambda x: prepare_dataset(
                x, 
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                root_path=self.config_dict['data']['test_root_path'],
            ),
            # remove_columns=[item for item in list(next(iter(dataset['test'])).keys()) if item not in ['language']]
        ).with_format('torch')

        # using kenlm for decoding with LM, required to sort the dictionary values as pyctcdecode vocab is positional dependent
        vocab_dict = dict(sorted(tokenizer.get_vocab().items(), key=lambda x:x[1]))
        vocab = list(vocab_dict.keys())
        vocab[vocab.index('[PAD]')] = '_'

        # build the decoder and load the kenlm langauge model
        decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=self.config_dict['kenlm']['model_path'],
            alpha=self.config_dict['kenlm']['alpha'], # weight associated with the LMs probabilities. A weight of 0 means the LM has no effect
            beta=self.config_dict['kenlm']['beta'], # weight associated with the number of words within the beam
        )

        # set up dataloader for the test set
        eval_dataloader = DataLoader(
            dataset["test"],
            batch_size=self.config_dict['model']['data_loader_batch_size'], 
            collate_fn=data_collator
        )

        decoded_preds_list, decoded_labels_list, decoded_preds_lm_list, decoded_labels_list_raw = [], [], [], []
        audio_filepath_list, language_list, duration_list = [], [], []

        model.eval() if not self.config_dict['model']['activate_dropout'] else model.train()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    input_features = batch["input_features"].to(device)
                    logits = model(input_features).logits
                    
                pred_ids = torch.argmax(logits, dim=-1)
                labels = batch["labels"]
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_pred_ids = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)

                decoded_pred_ids_lm_subwords = decoder.decode_batch(
                    pool=None, 
                    logits_list=logits.cpu().detach().numpy(),
                    beam_width=self.config_dict['kenlm']['beam_width'],
                    hotwords=self.config_dict['kenlm']['hotwords'],
                    hotword_weight=self.config_dict['kenlm']['hotword_weight'],
                )

                decoded_pred_ids_lm = [''.join(subword_list) for subword_list in [[tokenizer.decode(vocab_dict[token]) for token in entry] for entry in decoded_pred_ids_lm_subwords]]

                # normalized output
                decoded_labels_normalized = [
                    TextPostProcessingManager(
                        label=self.config_dict['data']['data_label'],
                        language=language,
                    ).process_data(
                        text=text
                    ) for text, language in zip(decoded_labels, batch['language'])
                ]

                decoded_preds_normalized = [
                    TextPostProcessingManager(
                        label=self.config_dict['data']['data_label'],
                        language=language,
                    ).process_data(
                        text=text
                    ) for text, language in zip(decoded_pred_ids, batch['language'])
                ]

                decoded_preds_lm_normalized = [
                    TextPostProcessingManager(
                        label=self.config_dict['data']['data_label'],
                        language=language,
                    ).process_data(
                        text=text
                    ) for text, language in zip(decoded_pred_ids_lm, batch['language'])
                ]

                # logging.getLogger('INFO').info(f'Ground Truth: {decoded_labels}\n')
                # logging.getLogger('INFO').info(f'Predicted (no LM): {decoded_pred_ids}\n')
                # logging.getLogger('INFO').info(f'Predicted (with LM): {decoded_pred_ids_lm}\n\n')

                # decoded_labels_list.extend(decoded_labels)
                # decoded_preds_list.extend(decoded_pred_ids)
                # decoded_preds_lm_list.extend(decoded_pred_ids_lm)

                logging.getLogger('INFO').info(f'Ground Truth: {decoded_labels_normalized}\n')
                logging.getLogger('INFO').info(f'Predicted (no LM): {decoded_preds_normalized}\n')
                logging.getLogger('INFO').info(f'Predicted (with LM): {decoded_preds_lm_normalized}\n\n')

                decoded_labels_list_raw.extend(decoded_labels)
                decoded_labels_list.extend(decoded_labels_normalized)
                decoded_preds_list.extend(decoded_preds_normalized)
                decoded_preds_lm_list.extend(decoded_preds_lm_normalized)

                audio_filepath_list.extend(batch['audio_filepath'])
                language_list.extend(batch['language'])
                duration_list.extend(batch['duration'])

        get_wer = WER(
        predictions=decoded_preds_list,
        references=decoded_labels_list,
        )

        get_cer = CER(
            predictions=decoded_preds_list,
            references=decoded_labels_list,
        )

        get_mer = MER(
            predictions=decoded_preds_list,
            references=decoded_labels_list,
        )

        get_wer_lm = WER(
            predictions=decoded_preds_lm_list,
            references=decoded_labels_list,
        )

        get_cer_lm = CER(
            predictions=decoded_preds_lm_list,
            references=decoded_labels_list,
        )

        get_mer_lm = MER(
            predictions=decoded_preds_lm_list,
            references=decoded_labels_list,
        )

        logging.getLogger('INFO').info(f"Test WER: {get_wer.compute():.5f}")
        logging.getLogger('INFO').info(f"Test CER: {get_cer.compute():.5f}")
        logging.getLogger('INFO').info(f"Test Word Accuracy: {(1-get_mer.compute()):.5f}\n")
        logging.getLogger('INFO').info(f"Test WER - with LM: {get_wer_lm.compute():.5f}")
        logging.getLogger('INFO').info(f"Test CER - with LM: {get_cer_lm.compute():.5f}")
        logging.getLogger('INFO').info(f"Test Word Accuracy - with LM: {(1-get_mer_lm.compute()):.5f}\n")

        logging.getLogger('INFO').info('Writing the predictions and labels to a json file...')

        with open(self.config_dict['results']['output_pred_path'], 'w+', encoding='utf-8') as fw:
            for audio_filepath, ref_raw, ref_norm, pred_norm, pred_norm_lm, duration, language in tqdm(
                zip(
                    audio_filepath_list,
                    decoded_labels_list_raw,
                    decoded_labels_list,
                    decoded_preds_list,
                    decoded_preds_lm_list,
                    duration_list,
                    language_list
                )
            ):
                fw.write(
                    json.dumps(
                        {
                            'audio_filepath': audio_filepath,
                            'text_raw': ref_raw,
                            'text': ref_norm,
                            'pred_str': pred_norm,
                            'pred_str_with_lm': pred_norm_lm,
                            'duration': duration,
                            'language': language,
                        },
                        ensure_ascii=False
                    ) + '\n'
                )


    def __call__(self) -> None:
        return self.start_evaluation()