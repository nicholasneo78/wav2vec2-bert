import hydra
from tasks import FinetuneModel

# python3 src/finetuning.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to finetune the model
    '''

    _ = FinetuneModel(
        cfg=cfg,
        is_remote=False,
    )()

if __name__ == '__main__':
    main()
