from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from transformers import PreTrainedTokenizerFast

class BuildBPETokenizer:

    '''
    To build the BPE Tokenizer
    '''

    def __init__(
        self,
        input_text_file_dir: str,
        output_tokenizer_dir: str,
        vocab_size: int,
    ) -> None:
        
        self.input_text_file_dir = input_text_file_dir
        self.output_tokenizer_dir = output_tokenizer_dir
        self.vocab_size = vocab_size

    def train_tokenizer(self) -> Tokenizer:

        '''
        method to train a tokenizer using BpeTrainer
        '''

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=True,
            special_tokens=[
                "[UNK]", "[SEP]", "[PAD]", "[MASK]", "[CLS]"
            ]
        )

        tokenizer.train([self.input_text_file_dir], trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()

        return tokenizer

    def wrap_tokenizer(self) -> PreTrainedTokenizerFast:

        '''
        wrapper method that trains a tokenizer, then wrap the tokenizer with huggingface PreTrainedTokenizerFast class 
        '''

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.train_tokenizer(),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
        )

        return wrapped_tokenizer
    
    def export_tokenizer(self) -> None:

        '''
        export the HuggingFace wrapped tokenizer
        '''

        self.wrap_tokenizer().save_pretrained(self.output_tokenizer_dir)

    def export_custom_tokenizer(self) -> None:

        '''
        export the custom tokenizer instead of the huggingface wrapped ones
        '''

        tokenizer = self.train_tokenizer()
        tokenizer.save(self.output_tokenizer_dir)

        return self.output_tokenizer_dir

    def __call__(self) -> None:

        # return self.export_tokenizer()
        return self.export_custom_tokenizer()