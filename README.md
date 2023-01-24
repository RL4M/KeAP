# Knowledge-exploited Auto-encoder for Proteins (KeAP)

This repository contains an official implementation of KeAP, presented by our ICLR'23 paper titled [Protein Representation Learning via Knowledge Enhanced Primary Structure Reasoning](https://openreview.net/forum?id=VbCMhg7MRmj). KeAP effectively encodes knowledge into protein language models by learning to exploit Gene Ontology knowledge graphs for protein primary structure reasoning. Some code was borrowed from [OntoProtein](https://github.com/zjunlp/OntoProtein).

## ProteinKG25 Configuration
[ProteinKG25]((https://zjunlp.github.io/project/ProteinKG25/)) is a large-scale knowledge graph dataset with aligned descriptions and protein sequences respectively to GO terms and protein entities. This dataset is necessary for performing pre-training. You can follow the [instruction](./ProteinKG25.md) to configure ProteinKG25.

- python 3.7
  
### Requirements for pre-training
- python 3.7
- pytorch 1.9
- transformer 4.5.1+
- deepspeed 0.6.5
- lmdb

Following [OntoProtein](https://github.com/zjunlp/OntoProtein), we also make small changes to the `deepspeed.py` file under transformers library (❗required for pre-training).
The changes can be applied by running:
```shell
cp replace_code/deepspeed.py usr/local/lib/python3.7/dist-packages/transformers/deepspeed.py
```

### Environment for protein-related tasks
<span id="environment-for-protein-related-tasks"></span>
python3.7 / pytorch 1.9 / transformer 4.5.1+ / lmdb / tape_proteins / scikit-multilearn / PyYAML

Pytorch-Geometric is required for the PPI task. Detailed environment configurations for the PPI task can be found in [GNN-PPI](https://github.com/lvguofeng/GNN_PPI)

Since the `tape_proteins` library only implemented the `P@L` metric for the contact prediction task, we add the `P@L/5` and `P@L/2` metrics by adding functions in the `models/modeling_utils.py` file in the `tape_proteins` library.
The changes can be applied by running:
```shell
cp replace_code/tape/modeling_utils.py /usr/local/lib/python3.7/dist-packages/tape/models/modeling_utils.py
```

## Data preparation
<span id="data-preparation"></span>

### Downstream task data
<span id="downstream-task-data"></span>
The data for [TAPE](https://github.com/songlab-cal/tape) tasks and PPI task can be downloaded from [[link]](https://drive.google.com/file/d/1snEAixeRokQW0wrJxLWtNA7m8VrzXN5A/view?usp=sharing).
The data for [PROBE](https://github.com/kansil/PROBE) tasks can be downloaded from [[link]](https://drive.google.com/file/d/1Sy0ldh_0fhAPatffTYJ7CENp3pbZHfyu/view?usp=sharing)

## Pre-training
<span id="protein-pre-training-model"></span>
You can pre-training your own KeAP using ProteinKG25. Before pretraining, you need to download two pretrained model: 
- [ProtBERT](https://huggingface.co/Rostlab/prot_bert) for initializing the protein encoder. 
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) for extracting text features in Gene Onotology Annotations. 

The script to run pre-training can be found in `script/run_pretrain.sh` and the detailed arguments are all listed in `src/training_args.py`. You will need to change the paths in `script/run_pretrain.sh` (PRETRAIN_DATA_DIR, ENCODER_MODEL_PATH, TEXT_MODEL_PATH) before running the script.

## Downstream tasks
<span id="downsteam-tasks"></span>

We release the checkpoint of KeAP. [Download model](https://drive.google.com/file/d/1CZFV8DA4l9F74ias1fR8mHdf1grrjsNq/view?usp=sharing) to run downstream tasks.

❗NOTE: You will need to change some paths to the downstream data and extracted embeddings (PPI and PROBE tasks) before running the code.

### TAPE tasks
<span id="tape-tasks"></span>
Secondary structure prediction, contact prediction, remote homology detection, stability prediction, and Fluorescence are tasks from [TAPE](https://github.com/songlab-cal/tape).

We provide scripts for these tasks in `script/`. You can also utilize the running codes `run_downstream.py` , and write your shell files according to your need:

- `run_downstream.py`: support `{ss3, ss8, contact, remote_homology, fluorescence}` tasks;
- `run_stability.py`: support `stability` task;


An example of fine-tuning KeAP for contact prediction (`script/run_contact.sh`) is as follows:

```shell
bash sh run_main.sh \
      --model output/pretrained/KeAP20/encoder \
      --output_file contact-KeAP20 \
      --task_name contact \
      --do_train True \
      --epoch 5 \
      --optimizer AdamW \
      --per_device_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --eval_step 50 \
      --eval_batchsize 1 \
      --warmup_ratio 0.08 \
      --learning_rate 3e-5 \
      --seed 3 \
      --frozen_bert False
```

Arguments for the training and evalution script are as follows,

- `--task_name`: Specify which task to evaluate on, and now the script supports `{ss3, ss8, contact, remote_homology, fluorescence, stability}` tasks;
- `--model`: The name or path of a protein pre-trained checkpoint.
- `--output_file`: The path of the fine-tuned checkpoint saved.
- `--do_train`: Specify if you want to finetune the pretrained model on downstream tasks.
- `--epoch`: Epochs for training model.
- `--optimizer`: The optimizer to use, e.g., `AdamW`.
- `--per_device_batch_size`: Batch size per GPU.
- `--gradient_accumulation_steps`: The number of gradient accumulation steps.
- `--eval_step`: Number of steps to run evaluation on validation set.
- `--eval_batchsize`: Evaluation batch size.
- `--warmup_ratio`: Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
- `--learning_rate`: Learning rate for fine-tuning
- `--frozen_bert`: Specify if you want to froze the encoder in the pretrained model.

More detailed parameters can be found in `run_main.sh`.

**Note: the best checkpoint is saved in** `OUTPUT_DIR/`.

### PROBE tasks
<span id="PROBE-tasks"></span>
Semantic similarity inference and binding affinity estimation are tasks from [PROBE](https://github.com/kansil/PROBE). The codes for PROBE can be found in `src/benchmark/PROBE`.

To test KeAP on these tasks:
- First extract embeddings using pre-trained KeAP by running the `src/benchmark/PROBE/extract_embeddings.py` script. 
- Then change the paths in `src/benchmark/PROBE/bin/probe_config.yaml` 
- Finally, run `src/benchmark/PROBE/bin/PROBE.py`. 

Detailed instructions and explanations of outputs can be found in [PROBE](https://github.com/kansil/PROBE).

### PPI task
<span id="PPI-tasks"></span>
The code for the protein-protein interaction downstream task is from [GNN-PPI](https://github.com/lvguofeng/GNN_PPI). The codes for PPI can be found in `src/benchmark/GNN_PPI`.

To test KeAP for protein-protein interaction (PPI) prediction:
- First extract embeddings using pre-trained KeAP by running the `src/benchmark/GNN_PPI/extract_protein_embeddings.py` script.
- Run `src/benchmark/GNN_PPI/run.py`.
