from math import gamma
import os
import json
import copy
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.file_utils import ModelOutput
from transformers.utils import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled
from deepspeed import DeepSpeedEngine
from transformers import DistilBertConfig, BertForMaskedLM
from transformers import pipeline
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List
# from decoder import KnowledgeBertModel
from src.decoder import KnowledgeBertModel

# import logging

logger = logging.get_logger('pretrain_log')
# logger = logging.getLogger("pretrain")


DECODER_CONFIG_NAME = "config.json"
PROTEIN_CONFIG_NAME = "config.json"
PROTEIN_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'
DECODER_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'


class KeAPConfig:
    """
    contains configs for the decoder, and configs for the 
    """
    def __init__(self,**kwargs):
        self.use_desc = kwargs.pop('use_desc', True)
        self.num_relations = kwargs.pop('num_relations', None)
        self.num_go_terms = kwargs.pop('num_go_terms', None)
        self.num_proteins = kwargs.pop('num_proteins', None)


        self.protein_encoder_cls = kwargs.pop('protein_encoder_cls', None)
        self.go_encoder_cls = kwargs.pop('go_encoder_cls', None)

        #         config.decoder_config.use_desc = self.use_desc
        # config.decoder_config.use_desc = self.num_relations
        # config.decoder_config.use_desc = self.num_go_terms
        # config.decoder_config.use_desc = self.num_proteins
        # config.decoder_config.use_desc = self.protein_encoder_cls
        # config.decoder_config.use_desc = self.go_encoder_cls


        self.protein_model_config = None
        self.decoder_config = None

    def save_to_json_file(self, encoder_save_directory: os.PathLike, decoder_save_directory: os.PathLike):
        os.makedirs(encoder_save_directory, exist_ok=True)
        os.makedirs(decoder_save_directory, exist_ok=True)

        self.protein_model_config.save_pretrained(encoder_save_directory)
        self.decoder_config.save_pretrained(decoder_save_directory)

        logger.info(f'Encoder Configuration saved in {encoder_save_directory}')
        logger.info(f'Decoder Configuration saved in {decoder_save_directory}')

    @classmethod
    def from_json_file(cls, encoder_config_path: os.PathLike, decoder_config_path: os.PathLike):
        config = cls()
        config.protein_model_config = AutoConfig.from_pretrained(encoder_config_path)
        config.decoder_config = AutoConfig.from_pretrained(decoder_config_path)

        return config

@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None


@dataclass
class MaskedLMAndPFIOutput(ModelOutput):

    mlm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attention: Optional[Tuple[torch.FloatTensor]] = None
    go_attention: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None
    pos_pfi_logits: Optional[torch.FloatTensor] = None
    neg_pfi_logits: Optional[torch.FloatTensor] = None


# only use the last layer----- we can try using other layers
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # attention_mask = attention_mask.bool()
        # num_batch_size = attention_mask.size(0)
        # pooled_output = torch.stack([hidden_states[i, attention_mask[i, :], :].mean(dim=0) for i in range(num_batch_size)], dim=0)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        return pooled_output


class KnowledgeDecoder(BertPreTrainedModel):
    """
    Implementation of the full KeAP decoder
    """

    def __init__(self,decoder_config=None):
        super().__init__(decoder_config)

        # textbert for relation and GO feature extraction, all param.requires_grad = False
        textbert_config = AutoConfig.from_pretrained(decoder_config.text_model_path)
        self.textbert = BertModel.from_pretrained(decoder_config.text_model_path, output_hidden_states=True)
        for param in self.textbert.parameters():
            param.requires_grad = False

        # decoder
        self.config = decoder_config
        self.decoder = KnowledgeBertModel(decoder_config,add_pooling_layer=False)

        # linear layer to project features into the same dimension
        self.go_project = nn.Linear(textbert_config.hidden_size, self.config.hidden_size)
        self.relation_project = nn.Linear(textbert_config.hidden_size, self.config.hidden_size)

        self.text_feat_dim = textbert_config.hidden_size
        self.text_pooler = BertPooler(textbert_config)

        # mlm head and pooler
        self.mlm_cls = BertOnlyMLMHead(self.config)
        self.pooler = BertPooler(self.config)

        # pfi head, requires pooled outputs
        if decoder_config.use_pfi:
            self.pfi_cls = nn.Sequential(nn.Linear(self.config.hidden_size, 2), nn.Softmax(dim=-1))
      
    def forward(self, 
        relation_inputs,
        go_inputs,
        inputs_embeds=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_mlm=True):
        batch, protein_len, protein_embed_size = inputs_embeds.size()

        ### go feature extraction
        go_input_ids, go_attention_mask, go_token_type_ids = go_inputs
        go_out = self.textbert(go_input_ids,
                                    attention_mask=go_attention_mask,
                                    token_type_ids=go_token_type_ids,
                                    output_hidden_states=True,
                                    return_dict=True) # (batch,token len, feat_dim)  

        # hidden size (b,seqlen, 768)
        go_feat = torch.cat(tuple([go_out.hidden_states[i].unsqueeze(1) for i in [-4, -3, -2, -1]]), dim=1) # (b ,4, go len, hidden_dim)
        go_feat = torch.mean(go_feat,dim=1) # (b,go len, hidden_dim)

        go_feat = self.go_project(go_feat) #(batch,, go len, decoder hidden dim)

        
        ### relation feature extraction
        relation_input_ids, relation_attention_mask, relation_token_type_ids = relation_inputs
        relation_out = self.textbert(relation_input_ids,
                                    attention_mask=relation_attention_mask,
                                    token_type_ids=relation_token_type_ids,
                                    output_hidden_states=True,
                                    return_dict=True) # (batch,token len, feat_dim)


        relation_feat = torch.cat(tuple([relation_out.hidden_states[i].unsqueeze(1) for i in [-4, -3, -2, -1]]), dim=1) # (b ,4,relation len, hidden_dim)
        relation_feat = torch.mean(relation_feat,dim=1) # (b,relation len, hidden_dim)

        relation_feat = self.relation_project(relation_feat) #(batch,relation len, decoder hidden dim)'
        
        ### input embedding to decoder, mask stay the same as protbert
        out = self.decoder(inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            relation_hidden_states=relation_feat,
            relation_attention_mask=relation_attention_mask,
            go_hidden_states=go_feat,
            go_attention_mask=go_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        # output_seq = out.hidden_states[-1]
        output_seq = out[0] # last hidden layer

        mlm_prediction_scores = self.mlm_cls(output_seq)

        # pfi output
        pooler_output = self.pooler(output_seq)
        pfi_prediction=None
        if self.config.use_pfi:
            pfi_prediction = self.pfi_cls(pooler_output)

        out.pooler_output = pooler_output

        return (out,mlm_prediction_scores,pfi_prediction)


class KeAP(nn.Module):
    """
    Implementation of the KeAP model
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder_config = config.protein_model_config
        self.decoder_config = config.decoder_config
        # main protein encoder
        self.encoder=BertModel(self.encoder_config, add_pooling_layer=False)
  
        # decoder
        self.decoder = KnowledgeDecoder(self.decoder_config)

    def forward(self,
        protein_inputs: Tuple = None,
        pos_relation_inputs: Union[torch.Tensor, Tuple] = None,
        pos_go_tail_inputs: Union[torch.Tensor, Tuple] = None,
        neg_relation_inputs: Union[torch.Tensor, Tuple] = None,
        neg_go_tail_inputs: Union[torch.Tensor, Tuple] = None,
        use_pfi: bool = True,
        output_attentions: bool = False
        ):


        protein_input_ids, protein_attention_mask, protein_token_type_ids = protein_inputs
        protein_outputs = self.encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions
        )

        # prot_seq_embed = protein_outputs.hidden_states[-1]
        prot_seq_embed = protein_outputs[0]

        # Note only consider mlm_loss calculated from "positive knowledge"
        out, mlm_prediction_scores, pos_pfi_prediction = self.decoder(pos_relation_inputs, pos_go_tail_inputs,inputs_embeds=prot_seq_embed,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions)

        neg_pfi_prediction=None
        if use_pfi:
            out_neg, neg_mlm_prediction_scores, neg_pfi_prediction = self.decoder(neg_relation_inputs, neg_go_tail_inputs,inputs_embeds=prot_seq_embed,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions)

        return MaskedLMAndPFIOutput(
            mlm_loss=None,
            mlm_logits=mlm_prediction_scores,
            hidden_states=out.hidden_states,
            encoder_attention=protein_outputs.attentions,
            go_attention=out.attentions,
            pooler_output=out.pooler_output,
            pos_pfi_logits=pos_pfi_prediction,
            neg_pfi_logits=neg_pfi_prediction
        )


    def save_pretrained(self,save_directory: os.PathLike,state_dict: Optional[dict] = None,save_config: bool = True,
    ):
        encoder_save_directory = os.path.join(save_directory, 'encoder')
        decoder_save_directory = os.path.join(save_directory, 'decoder')

        self.encoder.save_pretrained(encoder_save_directory, save_config=save_config)
        self.decoder.save_pretrained(decoder_save_directory, save_config=save_config)

        logger.info(f'Encoder Model weights saved in {encoder_save_directory}')
        logger.info(f'Decoder Model weights saved in {decoder_save_directory}')

    @classmethod
    def from_pretrained(
        cls, 
        protein_model_path: os.PathLike, 
        text_model_path: os.PathLike,
        decoder_model_path: os.PathLike,
        model_args = None,
        training_args = None,
        **kwargs
    ):

        # Will feed the number of relations and entity.
        num_relations = kwargs.pop('num_relations', None)
        num_go_terms = kwargs.pop('num_go_terms', None)
        num_proteins = kwargs.pop('num_proteins', None)

        # 1 assign useful configs to decoder config
        kmae_config = KeAPConfig.from_json_file(protein_model_path, decoder_model_path)
        kmae_config.decoder_config.num_relations=num_relations,
        kmae_config.decoder_config.num_go_terms=num_go_terms,
        kmae_config.decoder_config.num_proteins=num_proteins,
        if training_args:
            kmae_config.decoder_config.use_desc=training_args.use_desc,
            kmae_config.decoder_config.use_pfi = training_args.use_pfi
        if model_args:
            kmae_config.decoder_config.go_encoder_cls=model_args.go_encoder_cls,
            kmae_config.decoder_config.protein_encoder_cls=model_args.protein_encoder_cls

        kmae_config.decoder_config.text_model_path = text_model_path
        
        # instantiate model. Note textbert in decoder is initialized in this step
        kmae_model = cls(config=kmae_config)

        # 2 load encoder model
        if kmae_model.decoder_config.protein_encoder_cls == 'bert':
            kmae_model.encoder = BertModel.from_pretrained(protein_model_path)
        else:
            raise NotImplementedError("Currently only support bert for encoder")

        # 3 load decoder model
        if kmae_model.decoder_config.go_encoder_cls[0] == 'bert':
            # if decoder state dict exists load decoder
            if os.path.exists(os.path.join(decoder_model_path,'pytorch_model.bin')):
                logger.info(f'Loading Decoder Model from {decoder_model_path}')
                print(f'Loading Decoder Model from {decoder_model_path}')
                kmae_model.decoder = KnowledgeDecoder.from_pretrained(decoder_model_path)

            # if decoder state dict does not exists (first time training)
            else:
                kmae_model.decoder.decoder = KnowledgeBertModel(kmae_config.decoder_config)
        else:
            raise NotImplementedError("Currently only support bert cls")
        
        kmae_model.eval()

        return kmae_model

@dataclass
class KeAPLoss:
    """
     Perform forward propagation and return loss for protein function inference

    for pfi task (default don't use):
        pfi_weight: weight of protein function inference loss
        num_protein_go_neg_sample: number of negative samples per positive sample  
    """
    def __init__(self,pfi_weight=1.0,num_protein_go_neg_sample=1,mlm_lambda=1.0):
        self.pfi_weight = pfi_weight
        self.mlm_lambda = mlm_lambda
        self.num_protein_go_neg_sample = num_protein_go_neg_sample
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(
        self,
        model: KeAP,
        use_desc: bool = False,
        use_seq: bool = True,
        use_pfi: bool = True,
        protein_go_inputs = None,
        **kwargs
    ):
        # get protein inputs
        protein_mlm_input_ids = protein_go_inputs['protein_input_ids']
        protein_mlm_attention_mask = protein_go_inputs['protein_attention_mask']
        protein_mlm_token_type_ids = protein_go_inputs['protein_token_type_ids']
        protein_input = (protein_mlm_input_ids,protein_mlm_attention_mask,protein_mlm_token_type_ids)
        
        protein_mlm_labels = protein_go_inputs['protein_labels']

        # relation inputs
        relation_ids = protein_go_inputs['relation_ids']
        relation_attention_mask = protein_go_inputs['relation_attention_mask']
        relation_token_type_ids = protein_go_inputs['relation_token_type_ids']
        relation_inputs = (relation_ids, relation_attention_mask, relation_token_type_ids)


        ## positive inputs
        positive = protein_go_inputs['positive']

        # get tail inputs
        positive_tail_input_ids = positive['tail_input_ids']
        positive_tail_attention_mask = positive['tail_attention_mask']
        positive_tail_token_type_ids = positive['tail_token_type_ids']

        positive_go_tail_inputs = positive_tail_input_ids
        if use_desc:
            positive_go_tail_inputs = (positive_tail_input_ids, positive_tail_attention_mask, positive_tail_token_type_ids)


        ## negative inputs
        negative_go_tail_inputs=None
        if use_pfi:
            negative = protein_go_inputs['negative']

            # get tail inputs
            negative_tail_input_ids = negative['tail_input_ids']
            negative_tail_attention_mask = negative['tail_attention_mask']
            negative_tail_token_type_ids = negative['tail_token_type_ids']

            negative_go_tail_inputs = negative_tail_input_ids
            if use_desc:
                negative_go_tail_inputs = (negative_tail_input_ids, negative_tail_attention_mask, negative_tail_token_type_ids)



        model_output = model(protein_inputs=protein_input, 
        pos_relation_inputs=relation_inputs, 
        pos_go_tail_inputs=positive_go_tail_inputs,
        neg_relation_inputs=relation_inputs,
        neg_go_tail_inputs=negative_go_tail_inputs,
        use_pfi=use_pfi
        )

        # mlm loss
        mlm_logits = model_output.mlm_logits
        batch, seq_len, vocab_size = mlm_logits.size()
        mlm_loss = self.loss_fn(mlm_logits.view(-1, vocab_size), protein_mlm_labels.view(-1)) * self.mlm_lambda

        # pfi loss
        pos_pfi_loss =0
        neg_pfi_loss =0
        if use_pfi:
            pos_pfi_logits = model_output.pos_pfi_logits #(batch,2)
            neg_pfi_logits = model_output.neg_pfi_logits
   
            pos_pfi_label = protein_go_inputs['pfi_pos'].repeat(pos_pfi_logits.size(0))
            neg_pfi_label = protein_go_inputs['pfi_neg'].repeat(neg_pfi_logits.size(0))

            pos_pfi_loss = self.loss_fn(pos_pfi_logits.view(-1, 2), pos_pfi_label.view(-1)) * self.pfi_weight
            neg_pfi_loss = self.loss_fn(neg_pfi_logits.view(-1, 2), neg_pfi_label.view(-1)) * self.pfi_weight

        return(mlm_loss,pos_pfi_loss,neg_pfi_loss)

                
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


# performs pooling that do not considers pads efficiently, supports max,avg and summation
def pool(h, mask, type='max'):
    # h dim (batch,seq len, feat dim); mask dim(batch, seq len,1|feat dim)
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def copy_layers(src_layers, dest_layers, layers_to_copy):
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())