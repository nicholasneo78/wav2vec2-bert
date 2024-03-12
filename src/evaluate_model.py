import hydra
from tasks import EvaluateModel

# python3 src/evaluate_model.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to do model evaluation
    '''

    _ = EvaluateModel(
        cfg=cfg,
        is_remote=False,
    )()

if __name__ == '__main__':
    main()
