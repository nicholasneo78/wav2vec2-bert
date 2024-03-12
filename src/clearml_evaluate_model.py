from clearml import Task, Dataset, Model
import hydra

# python3 src/clearml_evaluate_model.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/clearml', config_name=None)
def main(cfg) -> None:

    '''
    main function call to do model evaluation
    '''

    args = cfg.evaluate_model

    task = Task.init(
        project_name=cfg.clearml_config.project_name, 
        task_name=args.clearml.task_name, 
        output_uri=cfg.clearml_config.output_url, 
    )

    task.set_base_docker(
        docker_image=cfg.clearml_config.docker_image,
    )

    task.execute_remotely(queue_name=args.clearml.queue, exit_process=True)

    from tasks import EvaluateModel
    import os
    import logging
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    os.makedirs('root/', exist_ok=True)

    dataset_test_path = Dataset.get(dataset_id=args.data.test.input_dataset_task_id).get_local_copy()
    tokenizer_path = Dataset.get(dataset_id=args.tokenizer.tokenizer_task_id).get_local_copy()
    finetuned_model_task_id = Task.get_task(task_id=args.model.finetuned_model_task_id).output_models_id
    finetuned_model_path = Model(model_id=finetuned_model_task_id['wav2vec2_bert_finetuned.zip']).get_local_copy()
    kenlm_model_path = Dataset.get(dataset_id=args.kenlm.lm_task_id).get_local_copy()

    cfg.evaluate_model.temp.dataset_test_path = dataset_test_path
    cfg.evaluate_model.temp.tokenizer_path = tokenizer_path 
    cfg.evaluate_model.temp.lm_path = kenlm_model_path

    # unzip the model directory
    shutil.unpack_archive(finetuned_model_path, os.path.dirname(finetuned_model_path))
    cfg.evaluate_model.temp.model_finetuned_path = os.path.dirname(finetuned_model_path)

    _ = EvaluateModel(
        cfg=cfg,
        is_remote=True,
    )()

    # create the dataset folder to store the manifest
    clearml_dataset = Dataset.create(
        dataset_project=cfg.clearml_config.dataset_project,
        dataset_name=args.results.output_dataset_name,
    )

    # save artifacts
    clearml_dataset_task = Task.get_task(clearml_dataset.id)

    # upload artifacts and files
    clearml_dataset_task.upload_artifact(
        name='results.json',
        artifact_object=args.results.output_pred_path,
    )

    clearml_dataset.add_files(path=args.results.output_pred_path, local_base_folder='root/')

    clearml_dataset.upload(output_url=cfg.clearml_config.output_url)
    clearml_dataset.finalize()

    logging.getLogger('INFO').info(f"ClearML Dataset ID - Test: {clearml_dataset.id}")
    logging.getLogger('INFO').info('Done')


if __name__ == '__main__':
    main()
