from clearml import Task, Dataset
import hydra

# python3 src/clearml_bpe_preprocessing.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/clearml', config_name=None)
def main(cfg) -> None:

    '''
    main function call to process the train and dev data then outputs the sentences.txt, subwords.txt and the tokenizer file for clearml
    '''

    args = cfg.bpe_preprocessing

    task = Task.init(
        project_name=cfg.clearml_config.project_name, 
        task_name=args.clearml.task_name, 
        output_uri=cfg.clearml_config.output_url, 
    )

    task.set_base_docker(
        docker_image=cfg.clearml_config.docker_image,
    )

    task.execute_remotely(queue_name=args.clearml.queue, exit_process=True)

    from tasks import BPEPreprocessing
    import os
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    os.makedirs('root/', exist_ok=True)

    dataset_train_path = Dataset.get(
        dataset_id=args.train.input_dataset_task_id,
    ).get_local_copy()

    dataset_dev_path = Dataset.get(
        dataset_id=args.dev.input_dataset_task_id,
    ).get_local_copy()

    # initalize cfg temp values
    cfg.bpe_preprocessing.temp.dataset_train_path = dataset_train_path
    cfg.bpe_preprocessing.temp.dataset_dev_path = dataset_dev_path

    sentence_path, bpe_tokenizer_path, bpe_corpus_path = BPEPreprocessing(
        cfg=cfg,
        is_remote=True
    )()

    logging.getLogger('INFO').info(f'Sentence path: {sentence_path}')
    logging.getLogger('INFO').info(f'BPE tokenizer path: {bpe_tokenizer_path}')
    logging.getLogger('INFO').info(f'BPE corpus path: {bpe_corpus_path}')

    clearml_dataset = Dataset.create(
        dataset_project=cfg.clearml_config.dataset_project,
        dataset_name=args.output_dataset_name,
    )

    # save artifacts
    clearml_dataset_task = Task.get_task(clearml_dataset.id)

    # upload artifacts and files
    clearml_dataset_task.upload_artifact(
        name='sentences.txt',
        artifact_object=args.sentences_text_path,
    )

    clearml_dataset_task.upload_artifact(
        name='tokenizer.json',
        artifact_object=args.tokenizer_path,
    )

    clearml_dataset_task.upload_artifact(
        name='subwords.txt',
        artifact_object=args.subwords_text_path,
    )

    clearml_dataset.add_files(path=sentence_path, local_base_folder='root/')
    clearml_dataset.add_files(path=bpe_tokenizer_path, local_base_folder='root/')
    clearml_dataset.add_files(path=bpe_corpus_path, local_base_folder='root/')

    clearml_dataset.upload(output_url=cfg.clearml_config.output_url)
    clearml_dataset.finalize()

    logging.getLogger('INFO').info(f"ClearML Dataset ID - Test: {clearml_dataset.id}")
    logging.getLogger('INFO').info('Done')

if __name__ == '__main__':
    main()