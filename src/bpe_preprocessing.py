import hydra
from tasks import BPEPreprocessing

# python3 src/bpe_preprocessing.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to process the train and dev data then outputs the sentences.txt, subwords.txt and the tokenizer file
    '''

    _ = BPEPreprocessing(
        cfg=cfg, 
        is_remote=False
    )()

if __name__ == '__main__':
    main()