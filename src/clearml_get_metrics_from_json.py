from clearml import Task, Dataset, Model
import hydra

# python3 src/clearml_get_metrics_from_json.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/clearml', config_name=None)
def main(cfg) -> None:

    '''
    main function call to get the metrics of the test set
    '''

    args = cfg.get_metrics_from_json
    
    task = Task.init(
        project_name=cfg.clearml_config.project_name, 
        task_name=args.clearml.task_name, 
        output_uri=cfg.clearml_config.output_url, 
    )

    task.set_base_docker(
        docker_image=cfg.clearml_config.docker_image,
    )

    task.execute_remotely(queue_name=args.clearml.queue, exit_process=True)

    from tasks import GetMetricsFromJSON
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    manifest_path = Dataset.get(dataset_id=args.manifest.manifest_task_id).get_local_copy()
    cfg.get_metrics_from_json.temp.manifest_path = manifest_path

    _ = GetMetricsFromJSON(
        cfg=cfg,
        is_remote=True,
    )()

    logging.getLogger('STATUS').info('DONE!')

if __name__ == '__main__':
    main()
