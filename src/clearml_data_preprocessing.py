from clearml import Task, Dataset
import hydra

# python3 src/clearml_data_preprocessing.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/clearml', config_name=None)
def main(cfg) -> None:

    '''
    main function call to convert the nemo manifest to the huggingface manifest for clearml
    '''

    args = cfg.data_preprocessing

    task = Task.init(
        project_name=cfg.clearml_config.project_name, 
        task_name=args.clearml.task_name, 
        output_uri=cfg.clearml_config.output_url, 
    )

    task.set_base_docker(
        docker_image=cfg.clearml_config.docker_image,
    )

    task.execute_remotely(queue_name=args.clearml.queue, exit_process=True)

    from tasks import BuildHuggingFaceDataManifest
    import os
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    os.makedirs('root/', exist_ok=True)

    if cfg.data_preprocessing.do_split:

        dataset_train_path = Dataset.get(
            dataset_id=args.train.input_dataset_task_id
        ).get_local_copy()

        dataset_dev_path = Dataset.get(
            dataset_id=args.dev.input_dataset_task_id
        ).get_local_copy()

        dataset_test_path = Dataset.get(
            dataset_id=args.test.input_dataset_task_id
        ).get_local_copy()

        # calling the base class to execute the logic
        df_train, path_train = BuildHuggingFaceDataManifest(
            input_manifest_path=os.path.join(dataset_train_path, args.train.input_manifest_path),
            output_manifest_path=args.train.output_manifest_path,
            data_label=cfg.sub_label,
        )()

        df_dev, path_dev = BuildHuggingFaceDataManifest(
            input_manifest_path=os.path.join(dataset_dev_path, args.dev.input_manifest_path),
            output_manifest_path=args.dev.output_manifest_path,
            data_label=cfg.sub_label,
        )()

        df_test, path_test = BuildHuggingFaceDataManifest(
            input_manifest_path=os.path.join(dataset_test_path, args.test.input_manifest_path),
            output_manifest_path=args.test.output_manifest_path,
            data_label=cfg.sub_label,
        )()

        logging.getLogger('INFO').info(f'Train path: {path_train}')
        logging.getLogger('INFO').info(f'Dev path: {path_dev}')
        logging.getLogger('INFO').info(f'Test path: {path_test}')

        dataset_train = Dataset.create(
            dataset_project=cfg.clearml_config.dataset_project,
            dataset_name=args.train.output_dataset_name,
            parent_datasets=[args.train.input_dataset_task_id],
        )

        dataset_dev = Dataset.create(
            dataset_project=cfg.clearml_config.dataset_project,
            dataset_name=args.dev.output_dataset_name,
            parent_datasets=[args.dev.input_dataset_task_id],
        )

        dataset_test = Dataset.create(
            dataset_project=cfg.clearml_config.dataset_project,
            dataset_name=args.test.output_dataset_name,
            parent_datasets=[args.test.input_dataset_task_id],
        )

        # save artifacts
        dataset_train_task = Task.get_task(dataset_train.id)
        dataset_dev_task = Task.get_task(dataset_dev.id)
        dataset_test_task = Task.get_task(dataset_test.id)

        # upload artifacts and files
        dataset_train_task.upload_artifact(name='train.json', artifact_object=args.train.output_manifest_path)
        dataset_dev_task.upload_artifact(name='dev.json', artifact_object=args.dev.output_manifest_path)
        dataset_test_task.upload_artifact(name='test.json', artifact_object=args.test.output_manifest_path)

        dataset_train.add_files(path=path_train, local_base_folder='root/')
        dataset_dev.add_files(path=path_dev, local_base_folder='root/')
        dataset_test.add_files(path=path_test, local_base_folder='root/')

        dataset_train.upload(output_url=cfg.clearml_config.output_url)
        dataset_train.finalize()
        dataset_dev.upload(output_url=cfg.clearml_config.output_url)
        dataset_dev.finalize()
        dataset_test.upload(output_url=cfg.clearml_config.output_url)
        dataset_test.finalize()

        # get the created dataset id
        logging.getLogger('INFO').info(f"ClearML Dataset ID - Train: {dataset_train.id}")
        logging.getLogger('INFO').info(f"ClearML Dataset ID - Dev: {dataset_dev.id}")
        logging.getLogger('INFO').info(f"ClearML Dataset ID - Test: {dataset_test.id}")

        logging.getLogger('INFO').info('Done')
    
    else:

        dataset_test_path = Dataset.get(
            dataset_id=args.test.input_dataset_task_id
        ).get_local_copy()

        df_test, path_test = BuildHuggingFaceDataManifest(
            input_manifest_path=os.path.join(dataset_test_path, args.test.input_manifest_path),
            output_manifest_path=args.test.output_manifest_path,
            data_label=cfg.sub_label,
        )()

        logging.getLogger('INFO').info(f'Test path: {path_test}')

        dataset_test = Dataset.create(
            dataset_project=cfg.clearml_config.dataset_project,
            dataset_name=args.test.output_dataset_name,
            parent_datasets=[args.test.input_dataset_task_id],
        )

        # save artifacts
        dataset_test_task = Task.get_task(dataset_test.id)

        dataset_test_task.upload_artifact(name='test.json', artifact_object=args.test.output_manifest_path)

        dataset_test.add_files(path=path_test, local_base_folder='root/')

        dataset_test.upload(output_url=cfg.clearml_config.output_url)
        dataset_test.finalize()

        logging.getLogger('INFO').info(f"ClearML Dataset ID - Test: {dataset_test.id}")

        logging.getLogger('INFO').info('Done')

if __name__ == '__main__':
    main()