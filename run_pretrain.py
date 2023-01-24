import os
# import logging
import json
import torch.nn as nn
from transformers import HfArgumentParser, set_seed
from transformers import BertTokenizer, AutoTokenizer, logging
from transformers.trainer_pt_utils import get_parameter_names

from src.models import KeAP, KeAPConfig, KnowledgeDecoder
from src.trainer import KeAPTrainer
from src.sampling import negative_sampling_strategy
from src.dataset import ProteinSeqDataset, ProteinGoDataset
from src.dataloader import DataCollatorForGoGo, DataCollatorForLanguageModeling, DataCollatorForProteinGo
from src.training_args import KMAEModelArguments, DataArguments, KMAETrainingArguments

logger = logging.get_logger(__name__)
DEVICE = 'cuda'


def main():
    parser = HfArgumentParser((KMAETrainingArguments, DataArguments, KMAEModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # check output_dir
    os.makedirs(training_args.output_dir, exist_ok=True)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info(f"Training parameters: {training_args}")

    set_seed(training_args.seed)

    # default BertTokenizer
    if model_args.encoder_model_file_name:
        protein_tokenizer = BertTokenizer.from_pretrained(model_args.encoder_model_file_name)
    else:
        raise ValueError("Need provide protein tokenizer config path.")

    text_tokenizer = None
    if model_args.text_model_file_name and training_args.use_desc:
        text_tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_file_name)

    # Load dataset
    protein_seq_dataset = None
    protein_go_dataset = None
    go_go_dataset = None

    # negative sampling strategy
    negative_sampling_fn = negative_sampling_strategy[data_args.negative_sampling_fn]

    if data_args.model_protein_go_data:
        protein_go_dataset = ProteinGoDataset(
            data_dir=data_args.pretrain_data_dir,
            use_desc=training_args.use_desc,
            use_seq=training_args.use_seq,
            protein_tokenizer=protein_tokenizer,
            text_tokenizer=text_tokenizer,
            negative_sampling_fn=negative_sampling_fn,
            num_neg_sample=training_args.num_protein_go_neg_sample,
            sample_head=data_args.protein_go_sample_head,
            sample_tail=data_args.protein_go_sample_tail,
            max_protein_seq_length=data_args.max_protein_seq_length,
            max_text_seq_length=data_args.max_text_seq_length
        )

    # whether to use protein function inference task during pretraining
    use_pfi = training_args.use_pfi

    # Ontology statistics
    num_relations = protein_go_dataset.num_relations
    num_go_terms = protein_go_dataset.num_go_terms
    num_proteins = protein_go_dataset.num_proteins

    # init data collator
    are_protein_length_same = False
    protein_seq_data_collator = DataCollatorForLanguageModeling(tokenizer=protein_tokenizer, are_protein_length_same=are_protein_length_same)
    protein_go_data_collator = DataCollatorForProteinGo(protein_tokenizer=protein_tokenizer, text_tokenizer=text_tokenizer, are_protein_length_same=are_protein_length_same, use_pfi=use_pfi)
    go_go_data_collator = DataCollatorForGoGo(tokenizer=text_tokenizer)

    model = KeAP.from_pretrained(
        protein_model_path=model_args.encoder_model_file_name,
        text_model_path=model_args.text_model_file_name,
        decoder_model_path = model_args.decoder_model_file_name,
        model_args=model_args,
        training_args=training_args,
        num_relations=num_relations,
        num_go_terms=num_go_terms,
        num_proteins=num_proteins,
    )


    # prepare Trainer
    trainer = KeAPTrainer(
        model=model,
        args=training_args,
        protein_seq_dataset=protein_seq_dataset,
        protein_go_dataset=protein_go_dataset,
        go_go_dataset=go_go_dataset,
        protein_seq_data_collator=protein_seq_data_collator,
        protein_go_data_collator=protein_go_data_collator,
        go_go_data_collator=go_go_data_collator
    )

    # Pretraining
    if training_args.do_train:
        # add path to checkpoint here to resume training
        trainer.train()


if __name__ == "__main__":
    main()