from typing import Dict
import os

from modules import BuildSentences, BuildBPETokenizer, BuildBPECorpus

class BPEPreprocessing:

    """
    BPE wrapper class that generates the output annotation text file, the BPE tokenizer and the annotation file in subwords format
    """

    def __init__(
        self,
        cfg: Dict,
        is_remote: bool,
    ) -> None:
        
        args = cfg.bpe_preprocessing
        
        if not is_remote:
            self.config_dict = {
                'train_manifest_dir': args.train_manifest_dir,
                'dev_manifest_dir': args.dev_manifest_dir,
                'sentences_text_dir': args.sentences_text_dir,
                'tokenizer_dir': args.tokenizer_dir,
                'tokenizer_vocab_size': args.tokenizer_vocab_size,
                'subwords_text_dir': args.subwords_text_dir,
            }
        else:
            self.config_dict = {
                'train_manifest_dir': os.path.join(args.temp.dataset_train_path, args.data.train.input_manifest_path),
                'dev_manifest_dir': os.path.join(args.temp.dataset_dev_path, args.data.dev.input_manifest_path),
                'sentences_text_dir': args.sentences_text_path,
                'tokenizer_dir': args.tokenizer_path,
                'tokenizer_vocab_size': args.tokenizer_vocab_size,
                'subwords_text_dir': args.subwords_text_path,
            }

    def bpe_wrapper(self) -> None:
        
        """
        Wrap all the 3 classes for processing the required BPE output, need to run in sequence because the next step depends on the output from the previous step
        """

        # build the sentences text files
        sentence_path = BuildSentences(
            output_text_dir=self.config_dict['sentences_text_dir'],
            train_manifest_dir=self.config_dict['train_manifest_dir'],
            dev_manifest_dir=self.config_dict['dev_manifest_dir'],
        )()

        # build the BPE tokenizer
        bpe_tokenizer_path = BuildBPETokenizer(
            input_text_file_dir=self.config_dict['sentences_text_dir'],
            output_tokenizer_dir=self.config_dict['tokenizer_dir'],
            vocab_size=self.config_dict['tokenizer_vocab_size'],
        )()

        # build the subword corpus
        bpe_corpus_path = BuildBPECorpus(
            input_text_file_dir=self.config_dict['sentences_text_dir'],
            input_tokenizer_dir=self.config_dict['tokenizer_dir'],
            output_text_file_dir=self.config_dict['subwords_text_dir'],
        )()

        return sentence_path, bpe_tokenizer_path, bpe_corpus_path

    def __call__(self) -> None:
        return self.bpe_wrapper()