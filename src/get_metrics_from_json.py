import hydra
from tasks import GetMetricsFromJSON

# python3 src/get_metrics_from_json.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to get the metrics of the data subset
    '''

    _ = GetMetricsFromJSON(
        cfg=cfg,
        is_remote=False,
    )()

if __name__ == '__main__':
    main()
