from typing import Dict
import os
from transformers.integrations import TensorBoardCallback
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, TrainingArguments, Trainer, PreTrainedTokenizerFast
import logging

from modules import load_huggingface_manifest, prepare_dataset, compute_metrics, DataCollatorCTCWithPadding, AdamW_constant_LR, AdamW_grouped_LLRD, SchedulerManager

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
    datefmt='%H:%M:%S'
)

class FinetuneModel:

    """
    Loads the dataset manifest Huggingface format and starts the finetuning
    """

    def __init__(self, cfg: Dict, is_remote: bool) -> None:
        
        args = cfg.finetuning
        
        if not is_remote:

            self.config_dict = {
                'data': {
                    'train_manifest_dir': args.data.train_manifest_dir,
                    'dev_manifest_dir': args.data.dev_manifest_dir,
                    'tokenizer_path': args.data.tokenizer_path,
                    'max_token_size': args.data.max_token_size,
                    'train_root_path': args.data.root_path,
                    'dev_root_path': args.data.root_path,
                },
                'model': {
                    'use_safetensors': args.model.use_safetensors,
                    'pretrained_model_dir': args.model.pretrained_model_dir,
                    'finetuned_output_dir': args.model.finetuned_output_dir,
                    'optimizer': {
                        'is_variable_lr': args.model.optimizer.is_variable_lr,
                        'lr_multiplier': args.model.optimizer.lr_multiplier
                    },
                    'scheduler': {
                        'type': args.model.scheduler.type,
                    },
                    'resume_from_checkpoint': args.model.resume_from_checkpoint,
                },
                'hyperparams': {
                    'attention_dropout': args.hyperparams.attention_dropout,
                    'hidden_dropout': args.hyperparams.hidden_dropout,
                    'feat_proj_dropout': args.hyperparams.feat_proj_dropout,
                    'mask_time_prob': args.hyperparams.mask_time_prob,
                    'final_dropout': args.hyperparams.final_dropout,
                    'layerdrop': args.hyperparams.layerdrop,
                    'ctc_loss_reduction': args.hyperparams.ctc_loss_reduction,
                    'add_adapter': args.hyperparams.add_adapter,
                    'num_adapter_layers': args.hyperparams.num_adapter_layers,
                    'use_intermediate_ffn_before_adapter': args.hyperparams.use_intermediate_ffn_before_adapter,
                    'num_train_epochs': args.hyperparams.num_train_epochs,
                    'per_device_train_batch_size': args.hyperparams.per_device_train_batch_size,
                    'per_device_eval_batch_size': args.hyperparams.per_device_eval_batch_size,
                    'gradient_accumulation_steps': args.hyperparams.gradient_accumulation_steps,
                    'lr': args.hyperparams.lr,
                    'weight_decay': args.hyperparams.weight_decay,
                    'warmup_ratio': args.hyperparams.warmup_ratio,
                    'gradient_checkpointing': args.hyperparams.gradient_checkpointing,
                    'fp16': args.hyperparams.fp16,
                    'save_epoch_interval': args.hyperparams.save_epoch_interval,
                    'eval_epoch_interval': args.hyperparams.eval_epoch_interval,
                    'logging_epoch_interva1': args.hyperparams.logging_epoch_interva1,
                    'save_total_limit': args.hyperparams.save_total_limit,
                    'save_safetensors': args.hyperparams.save_safetensors,
                    'ignore_data_skip': args.hyperparams.ignore_data_skip,
                    'greater_is_better': args.hyperparams.greater_is_better,
                    'load_best_model_at_end': args.hyperparams.load_best_model_at_end,
                    'metric_for_best_model': args.hyperparams.metric_for_best_model,
                }
            }

        else:

            self.config_dict = {
                'data': {
                    'train_manifest_dir': os.path.join(args.temp.dataset_train_path, args.data.train.input_manifest_path),
                    'dev_manifest_dir': os.path.join(args.temp.dataset_dev_path, args.data.dev.input_manifest_path),
                    'tokenizer_path': os.path.join(args.temp.tokenizer_path, args.tokenizer.tokenizer_path),
                    'max_token_size': args.tokenizer.max_token_size,
                    'train_root_path': args.temp.dataset_train_path,
                    'dev_root_path': args.temp.dataset_dev_path,
                },
                'model': {
                    'use_safetensors': args.model.use_safetensors,
                    'pretrained_model_dir': os.path.join(args.temp.model_pretrained_path, args.model.input_pretrained_model_path),
                    'finetuned_output_dir': args.model.output_finetuned_model_path,
                    'optimizer': {
                        'is_variable_lr': args.model.optimizer.is_variable_lr,
                        'lr_multiplier': args.model.optimizer.lr_multiplier
                    },
                    'scheduler': {
                        'type': args.model.scheduler.type,
                    },
                    'resume_from_checkpoint': args.model.resume_from_checkpoint,
                },
                'hyperparams': {
                    'attention_dropout': args.hyperparams.attention_dropout,
                    'hidden_dropout': args.hyperparams.hidden_dropout,
                    'feat_proj_dropout': args.hyperparams.feat_proj_dropout,
                    'mask_time_prob': args.hyperparams.mask_time_prob,
                    'final_dropout': args.hyperparams.final_dropout,
                    'layerdrop': args.hyperparams.layerdrop,
                    'ctc_loss_reduction': args.hyperparams.ctc_loss_reduction,
                    'add_adapter': args.hyperparams.add_adapter,
                    'num_adapter_layers': args.hyperparams.num_adapter_layers,
                    'use_intermediate_ffn_before_adapter': args.hyperparams.use_intermediate_ffn_before_adapter,
                    'num_train_epochs': args.hyperparams.num_train_epochs,
                    'per_device_train_batch_size': args.hyperparams.per_device_train_batch_size,
                    'per_device_eval_batch_size': args.hyperparams.per_device_eval_batch_size,
                    'gradient_accumulation_steps': args.hyperparams.gradient_accumulation_steps,
                    'lr': args.hyperparams.lr,
                    'weight_decay': args.hyperparams.weight_decay,
                    'warmup_ratio': args.hyperparams.warmup_ratio,
                    'gradient_checkpointing': args.hyperparams.gradient_checkpointing,
                    'fp16': args.hyperparams.fp16,
                    'save_epoch_interval': args.hyperparams.save_epoch_interval,
                    'eval_epoch_interval': args.hyperparams.eval_epoch_interval,
                    'logging_epoch_interva1': args.hyperparams.logging_epoch_interva1,
                    'save_total_limit': args.hyperparams.save_total_limit,
                    'save_safetensors': args.hyperparams.save_safetensors,
                    'ignore_data_skip': args.hyperparams.ignore_data_skip,
                    'greater_is_better': args.hyperparams.greater_is_better,
                    'load_best_model_at_end': args.hyperparams.load_best_model_at_end,
                    'metric_for_best_model': args.hyperparams.metric_for_best_model,
                }
            }

    def filter_labels(self, labels_length) -> bool:

        """
        Filter empty label sequences
        """

        return 0 < len(labels_length) < self.config_dict['data']['max_token_size']
    
    def start_finetuning(self) -> None:

        """
        main method to load the manifest file, grabs the data and send the data for finetuning
        """

        
        dataset, train_data_len = load_huggingface_manifest(
            train_dir=self.config_dict['data']['train_manifest_dir'],
            dev_dir=self.config_dict['data']['dev_manifest_dir'],
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.config_dict['data']['tokenizer_path'],
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]"
        )

        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(self.config_dict['model']['pretrained_model_dir'])

        data_collator = DataCollatorCTCWithPadding(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            padding=True
        )

        dataset['train'] = dataset['train'].map(
            lambda x: prepare_dataset(
                x, 
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                root_path=self.config_dict['data']['train_root_path'],
            ),
        ).with_format('torch')

        dataset['dev'] = dataset['dev'].map(
            lambda x: prepare_dataset(
                x, 
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                root_path=self.config_dict['data']['dev_root_path'],
            ),
        ).with_format('torch')

        # filter ground truth transcriptions that exceeds the max token length of the data point, if not longer sentences will throw errors
        dataset = dataset.filter(self.filter_labels, input_columns=['labels'])

        # shuffle the train and dev set
        dataset['train'] = dataset['train'].shuffle(seed=0)

        model = Wav2Vec2BertForCTC.from_pretrained(
            self.config_dict['model']['pretrained_model_dir'],
            use_safetensors=self.config_dict['model']['use_safetensors'],
            attention_dropout=self.config_dict['hyperparams']['attention_dropout'],
            hidden_dropout=self.config_dict['hyperparams']['hidden_dropout'],
            feat_proj_dropout=self.config_dict['hyperparams']['feat_proj_dropout'],
            mask_time_prob=self.config_dict['hyperparams']['mask_time_prob'],
            final_dropout=self.config_dict['hyperparams']['final_dropout'],
            layerdrop=self.config_dict['hyperparams']['layerdrop'],
            ctc_loss_reduction=self.config_dict['hyperparams']['ctc_loss_reduction'],
            add_adapter=self.config_dict['hyperparams']['add_adapter'],
            num_adapter_layers=self.config_dict['hyperparams']['num_adapter_layers'],
            use_intermediate_ffn_before_adapter=self.config_dict['hyperparams']['use_intermediate_ffn_before_adapter'],
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=len(tokenizer),
        )

        max_train_steps = train_data_len * self.config_dict['hyperparams']['num_train_epochs'] // (self.config_dict['hyperparams']['per_device_train_batch_size'] * self.config_dict['hyperparams']['gradient_accumulation_steps']) + 1

        training_args = TrainingArguments(
            output_dir=self.config_dict['model']['finetuned_output_dir'],
            learning_rate=self.config_dict['hyperparams']['lr'],
            weight_decay=self.config_dict['hyperparams']['weight_decay'],
            warmup_ratio=self.config_dict['hyperparams']['warmup_ratio'],
            per_device_train_batch_size=self.config_dict['hyperparams']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config_dict['hyperparams']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config_dict['hyperparams']['gradient_accumulation_steps'],
            gradient_checkpointing=self.config_dict['hyperparams']['gradient_checkpointing'],
            gradient_checkpointing_kwargs={'use_reentrant': True},
            fp16=self.config_dict['hyperparams']['fp16'],
            evaluation_strategy="steps",
            max_steps=max_train_steps, 
            save_steps=int(self.config_dict['hyperparams']['save_epoch_interval'] / self.config_dict['hyperparams']['num_train_epochs'] * max_train_steps) + 1,
            eval_steps=int(self.config_dict['hyperparams']['eval_epoch_interval'] / self.config_dict['hyperparams']['num_train_epochs'] * max_train_steps) + 1,
            logging_steps=int(self.config_dict['hyperparams']['logging_epoch_interva1'] / self.config_dict['hyperparams']['num_train_epochs'] * max_train_steps) + 1,
            save_total_limit=self.config_dict['hyperparams']['save_total_limit'],
            push_to_hub=False,
            report_to='tensorboard',
            save_safetensors=self.config_dict['hyperparams']['save_safetensors'],
            ignore_data_skip=self.config_dict['hyperparams']['ignore_data_skip'],
            greater_is_better=self.config_dict['hyperparams']['greater_is_better'],
            load_best_model_at_end=self.config_dict['hyperparams']['load_best_model_at_end'],
            metric_for_best_model=self.config_dict['hyperparams']['metric_for_best_model'],
        )

        optimizer = AdamW_grouped_LLRD(
            model=model,
            init_lr=training_args.learning_rate, 
            weight_decay=training_args.weight_decay,
            multiplier=self.config_dict['model']['optimizer']['lr_multiplier'],
        ) if self.config_dict['model']['optimizer']['is_variable_lr'] else AdamW_constant_LR(
            model=model,
            init_lr=training_args.learning_rate, 
            weight_decay=training_args.weight_decay,
        )

        scheduler = SchedulerManager(
            scheduler_name=self.config_dict['model']['scheduler']['type'],
            optimizer=optimizer,
            warmup_ratio=training_args.warmup_ratio,
            num_training_steps=training_args.max_steps,
        )()

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=lambda x: compute_metrics(
                x,
                tokenizer=tokenizer,
            ),
            train_dataset=dataset['train'],
            eval_dataset=dataset['dev'],
            tokenizer=tokenizer,
            callbacks=[
                    TensorBoardCallback()
                ],
            optimizers=(
                optimizer,
                scheduler,
            )
        )

        feature_extractor.save_pretrained(self.config_dict['model']['finetuned_output_dir'], safe_serialization=True)

        trainer.train(resume_from_checkpoint=self.config_dict['model']['resume_from_checkpoint'])
    
    def __call__(self) -> None:
        return self.start_finetuning()