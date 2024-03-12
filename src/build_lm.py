import hydra
from tasks import BuildLM

# python3 src/build_lm.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to build the language model
    '''

    _, _ = BuildLM(
        cfg=cfg,
        is_remote=False,
    )()

if __name__ == '__main__':
    main()
