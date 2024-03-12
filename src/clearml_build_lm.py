from clearml import Task, Dataset
import hydra

# python3 src/clearml_build_lm.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/clearml', config_name=None)
def main(cfg) -> None:

    '''
    main function call to build the language model
    '''

    args = cfg.build_lm

    task = Task.init(
        project_name=cfg.clearml_config.project_name, 
        task_name=args.clearml.task_name, 
        output_uri=cfg.clearml_config.output_url, 
    )

    task.set_base_docker(
        docker_image=cfg.clearml_config.docker_image,
    )

    task.execute_remotely(queue_name=args.clearml.queue, exit_process=True)

    from tasks import BuildLM
    import os
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    os.makedirs('root/', exist_ok=True)

    subwords_path = Dataset.get(
        dataset_id=args.data.text_file_path_id,
    ).get_local_copy()

    # initalize cfg temp values
    cfg.build_lm.temp.text_file_path = subwords_path

    raw_output_lm_path, corrected_output_lm_path = BuildLM(
        cfg=cfg,
        is_remote=True
    )()

    logging.getLogger('INFO').info(f'Raw output LM path: {raw_output_lm_path}')
    logging.getLogger('INFO').info(f'Corrected output LM path: {corrected_output_lm_path}')

    clearml_dataset = Dataset.create(
        dataset_project=cfg.clearml_config.dataset_project,
        dataset_name=args.output_dataset_name,
    )

    # save artifacts
    clearml_dataset_task = Task.get_task(clearml_dataset.id)

    # upload artifacts and files
    clearml_dataset_task.upload_artifact(
        name='lm.arpa',
        artifact_object=args.raw_output_lm_path,
    )

    clearml_dataset_task.upload_artifact(
        name='corrected_lm.arpa',
        artifact_object=args.corrected_output_lm_path,
    )

    clearml_dataset.add_files(path=raw_output_lm_path, local_base_folder='root/')
    clearml_dataset.add_files(path=corrected_output_lm_path, local_base_folder='root/')

    clearml_dataset.upload(output_url=cfg.clearml_config.output_url)
    clearml_dataset.finalize()

    logging.getLogger('INFO').info(f"ClearML Dataset ID - Test: {clearml_dataset.id}")
    logging.getLogger('INFO').info('Done')

if __name__ == '__main__':
    main()