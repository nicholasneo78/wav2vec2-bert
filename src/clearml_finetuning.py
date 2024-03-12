from clearml import Task, Dataset, Model
import hydra

# python3 src/clearml_finetuning.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/clearml', config_name=None)
def main(cfg) -> None:

    '''
    main function call to finetune the model
    '''

    args = cfg.finetuning

    task = Task.init(
        project_name=cfg.clearml_config.project_name, 
        task_name=args.clearml.task_name, 
        output_uri=cfg.clearml_config.output_url, 
    )

    task.set_base_docker(
        docker_image=cfg.clearml_config.docker_image,
    )

    task.execute_remotely(queue_name=args.clearml.queue, exit_process=True)

    from tasks import FinetuneModel
    import os
    import logging
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    os.makedirs('root/', exist_ok=True)

    dataset_train_path = Dataset.get(dataset_id=args.data.train.input_dataset_task_id).get_local_copy()
    dataset_dev_path = Dataset.get(dataset_id=args.data.dev.input_dataset_task_id).get_local_copy()
    tokenizer_path = Dataset.get(dataset_id=args.tokenizer.tokenizer_task_id).get_local_copy()

    logging.getLogger('INFO').info(f'Dataset train path: {dataset_train_path}')
    logging.getLogger('INFO').info(f'Items in the train path: {os.listdir(dataset_train_path)}')

    logging.getLogger('INFO').info(f'Tokenizer path: {tokenizer_path}')
    logging.getLogger('INFO').info(f'Items in the tokenizer path: {os.listdir(tokenizer_path)}')

    model_pretrained_path_zip = Model(model_id=args.model.pretrained_task_id)

    # extract the clearml model zip
    shutil.unpack_archive(model_pretrained_path_zip, os.path.dirname(model_pretrained_path_zip))
    model_pretrained_path = os.path.dirname(model_pretrained_path_zip)

    logging.getLogger('INFO').info(f'Model pretrained path: {model_pretrained_path}')
    logging.getLogger('INFO').info(f'Files in the model pretrained path: {os.listdir(model_pretrained_path)}')

    cfg.finetuning.temp.dataset_train_path = dataset_train_path
    cfg.finetuning.temp.dataset_dev_path = dataset_dev_path
    cfg.finetuning.temp.tokenizer_path = tokenizer_path
    cfg.finetuning.temp.model_pretrained_path = model_pretrained_path

    _ = FinetuneModel(
        cfg=cfg,
        is_remote=True,
    )()

    logging.getLogger('INFO').info('Compressing the model files into a zip file')
    shutil.make_archive(base_name=args.model.output_finetuned_model_path[:-1], format='zip', root_dir='.', base_dir=args.model.output_finetuned_model_path)

    # upload model as output model of the task
    task.update_output_model(
        model_path=f"{args.model.output_finetuned_model_path[:-1]}.zip",
        name="wav2vec2_bert_finetuned.zip",
        model_name="wav2vec2_bert_finetuned.zip",
    )

    logging.getLogger('STATUS').info(f"ClearML Finetuned Model ID: {task.id}")
    logging.getLogger('STATUS').info('DONE!')

    # close the task
    task.close()

if __name__ == '__main__':
    main()