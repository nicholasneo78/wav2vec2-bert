import hydra
from tasks import BuildHuggingFaceDataManifest

# python3 src/data_preprocessing.py --config-name <config_file>

@hydra.main(version_base=None, config_path='conf/local', config_name=None)
def main(cfg) -> None:

    '''
    main function call to convert the nemo manifest to the huggingface manifest 
    '''

    args = cfg.data_preprocessing

    if args.do_split:

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=args.split.input_manifest_path_train,
            output_manifest_path=args.split.output_manifest_path_train,
            data_label=args.data_label,
        )()

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=args.split.input_manifest_path_dev,
            output_manifest_path=args.split.output_manifest_path_dev,
            data_label=args.data_label,
        )()

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=args.split.input_manifest_path_test,
            output_manifest_path=args.split.output_manifest_path_test,
            data_label=args.data_label,
        )()

    else:

        _, _ = BuildHuggingFaceDataManifest(
            input_manifest_path=args.no_split.input_manifest_path,
            output_manifest_path=args.no_split.output_manifest_path,
            data_label=args.data_label,
        )()


if __name__ == '__main__':
    main()