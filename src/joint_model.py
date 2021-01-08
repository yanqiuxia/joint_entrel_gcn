#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/10/12 16:53:03

@author: Changzhi Sun
"""
import sys
sys.path.append("..")
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.word_encoder import WordCharEncoder
from modules.seq2seq_encoders.seq2seq_bilstm import BiLSTMEncoder
from src.seq_decoder import SeqSoftmaxDecoder
from src.decoder import VanillaSoftmaxDecoder
from lib.vocabulary import Vocabulary
from src.ent_span_generator import EntSpanGenerator
from src.rel_feat_extractor import RelFeatExtractor
from src.ent_span_feat_extractor import EntSpanFeatExtractor
from src.gcn_extractor import GCNExtractor
from src.graph_cnn_encoder import GCN


class JointModel(nn.Module):

    def __init__(self,
                 word_encoder: WordCharEncoder,
                 seq2seq_encoder: BiLSTMEncoder,
                 ent_span_decoder: SeqSoftmaxDecoder,
                 ent_span_feat_extractor: EntSpanFeatExtractor,
                 ent_ids_decoder: VanillaSoftmaxDecoder,
                 rel_feat_extractor: RelFeatExtractor,
                 rel_decoder: VanillaSoftmaxDecoder,
                 bin_rel_decoder: VanillaSoftmaxDecoder,
                 gcn: GCN,
                 vocab: Vocabulary,
                 sch_k: float,
                 use_cuda: bool,
                 max_entity_num:int) -> None:
        super(JointModel, self).__init__()
        self.word_encoder = word_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.ent_span_decoder = ent_span_decoder
        self.ent_span_feat_extractor = ent_span_feat_extractor
        self.ent_ids_decoder = ent_ids_decoder
        self.rel_feat_extractor = rel_feat_extractor
        self.rel_decoder = rel_decoder
        self.bin_rel_decoder = bin_rel_decoder
        self.vocab = vocab
        self.sch_k = sch_k
        self.use_cuda = use_cuda
        self.ent_span_generator = EntSpanGenerator(vocab,max_entity_num)
        self.gcn_extractor = GCNExtractor(gcn, use_cuda)

    def forward(self, batch: Dict[str, Any]) -> Dict:
        token_inputs = batch['tokens']
        # char_inputs = batch['token_chars']
        seq_tags = batch['ent_span_labels']
        seq_lens = batch['seq_lens']

        #1---提取wordembedding,采用bilstm编码,使用softmax输出ent_span_labels
        encoder_inputs = self.word_encoder(token_inputs) #(batch_size, sent_size, word_encoder_size)
        seq_feats = self.seq2seq_encoder(encoder_inputs, seq_lens).contiguous()#(batch_size,sequence_size,hidden_size)
        ent_span_outputs = self.ent_span_decoder(seq_feats, seq_tags)#[loss, (batch_size, seq_size)]

        ent_span_pred = ent_span_outputs['predict'] #[(batch_size, seq_size)]
        #2---随机选取真实和预测的标签进行训练，由sch_k系数控制，实验的时候使用真实标签
        if self.training and batch['i_epoch'] is not None:
            sch_p = self.sch_k / (self.sch_k + np.exp(batch['i_epoch'] / self.sch_k))
            # sch_p = 1.0
            ent_span_pred = [gold if np.random.random() < sch_p else pred
                             for pred, gold in zip(ent_span_pred, batch['ent_span_labels'])]
            ent_span_pred = torch.stack(ent_span_pred) #[(batch_size, seq_size)]
        #3---生成ent_ids, ent_ids_label。ent_ids由span组成，与batch有点不一样，end就是当前字符结束位置， ent_ids_label由实体标签id组成。
        all_ent_ids, all_ent_ids_label = self.ent_span_generator.forward(
            batch, ent_span_pred, self.training)
        batch['all_ent_ids'] = all_ent_ids  # (batch_size, ent_span_num, 2)
        batch['all_ent_ids_label'] = all_ent_ids_label # (batch_size, ent_span_num, 1)
        batch['seq_feats'] = seq_feats

        outputs = {}
        outputs['ent_span_loss'] = ent_span_outputs['loss']
        outputs['ent_span_pred'] = ent_span_pred

        ent_ids_num = sum([len(e) for e in batch['all_ent_ids']])

        if ent_ids_num == 0:
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.use_cuda:
                zero_loss = zero_loss.cuda(non_blocking=True)
            batch_size, seq_size = batch['tokens'].size()
            batch_tag = [[self.vocab.get_token_index("O", "ent_labels") for _ in range(seq_size)]
                         for _ in range(batch_size)]
            outputs['ent_ids_loss'] = zero_loss
            outputs['all_ent_pred'] = batch_tag
            outputs['rel_loss'] = zero_loss
            outputs['bin_rel_loss'] = zero_loss
            outputs['all_rel_pred'] = [[] for _ in range(batch_size)]
            outputs['all_bin_rel_pred'] = [[] for _ in range(batch_size)]
            outputs['all_candi_rels'] = [[] for _ in range(batch_size)]
            return outputs
        #4---获取所有候选关系,candi_rels由实体1和实体2的span组成， rel_labels为关系标签id
        all_candi_rels, all_rel_labels = self.generate_all_candidates(batch)
        batch['all_candi_rels'] = all_candi_rels
        batch['all_rel_labels'] = all_rel_labels

        candi_rel_num = sum([len(e) for e in all_candi_rels])

        #5---实体特征抽取，采用卷积提取span特征
        # ent_ids_span_feats:(ent_ids_num, hidden_size)
        ent_ids_batch, ent_ids_span_feats = self.ent_span_feat_extractor(batch)
        if candi_rel_num > 0:
            #6---关系特征抽取，采用卷积提取实体1,实体2,以及上下文5部分的特征
            rel_batch = self.rel_feat_extractor(batch, ent_ids_span_feats)
            assert candi_rel_num == rel_batch['inputs'].size(0)
            rel_batch['all_bin_rel_labels'] = (rel_batch['all_rel_labels'] != self.vocab.get_token_index("None", "rel_labels")).long()
            bin_rel_outputs = self.bin_rel_decoder(
                rel_batch['inputs'], rel_batch['all_bin_rel_labels'])
            bin_rel_probs = bin_rel_outputs['softmax_probs'][:, 1]
        else:
            rel_batch = None
            bin_rel_probs = None
        #8---实体和关系输入gcn，提取gcn特征
        all_ent_gcn_feats, all_rel_gcn_feats = self.gcn_extractor(
            batch, ent_ids_span_feats, rel_batch, bin_rel_probs)

        all_ent_feats = torch.cat([ent_ids_span_feats, all_ent_gcn_feats], 1)
        
        if candi_rel_num == 0:

            ent_ids_outputs = self.ent_ids_decoder(all_ent_feats, 
                                                   ent_ids_batch['all_ent_ids_label'])
            all_ent_pred = self.create_all_ent_pred(batch, ent_ids_outputs)

            outputs['ent_ids_loss'] = ent_ids_outputs['loss']
            outputs['all_ent_pred'] = all_ent_pred
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.use_cuda:
                zero_loss = zero_loss.cuda(non_blocking=True)

            batch_size, _ = batch['tokens'].size()
            outputs['rel_loss'] = zero_loss
            outputs['bin_rel_loss'] = zero_loss
            outputs['all_rel_pred'] = [[] for _ in range(batch_size)]
            outputs['all_bin_rel_pred'] = [[] for _ in range(batch_size)]
            outputs['all_candi_rels'] = [[] for _ in range(batch_size)]
            return outputs

        #9--利用实体特征输出提取实体节点
        ent_ids_outputs = self.ent_ids_decoder(all_ent_feats, 
                                               ent_ids_batch['all_ent_ids_label'])
        #9.1---生成实体预测的span以及span的标签
        all_ent_pred = self.create_all_ent_pred(batch, ent_ids_outputs)

        outputs['ent_ids_loss'] = ent_ids_outputs['loss']
        outputs['all_ent_pred'] = all_ent_pred

        # 10--利用关系特征输出提取关系节点
        all_rel_feats = torch.cat([rel_batch['inputs'], all_rel_gcn_feats], 1)
        rel_outputs = self.rel_decoder(all_rel_feats,
                                       rel_batch['all_rel_labels'])
        outputs['rel_loss'] = rel_outputs['loss']

        #10.1---生成所有关系，实体1和实体2是否有关系，以及所预测的关系标签
        all_candi_rels, all_rel_pred, all_bin_rel_pred = self.create_all_rel_pred(
            all_candi_rels, batch, ent_ids_outputs, rel_outputs, bin_rel_outputs)

        outputs['all_rel_pred'] = all_rel_pred
        outputs['all_bin_rel_pred'] = all_bin_rel_pred
        outputs['all_candi_rels'] = all_candi_rels
        outputs['bin_rel_loss'] = bin_rel_outputs['loss'] 
        return outputs

    def create_all_rel_pred(
            self, 
            all_candi_rels: List[List[Tuple[Tuple, Tuple]]],
            batch: Dict[str, Any],
            ent_ids_outputs: Dict[str, Any],
            rel_outputs: Dict[str, Any],
            bin_rel_outputs: Dict[str, Any]) -> (List[List[Tuple]], List[List[int]], List[List[int]]):
        ret_rel_pred = []
        ret_candi_rels = []
        ret_bin_rel_pred = []
        i_ent = 0
        i_rel = 0
        for i in range(len(all_candi_rels)):
            cur_candi_rels = all_candi_rels[i]
            
            cur_ent_ids = batch['all_ent_ids'][i]
            num_ent_ids = len(batch['all_ent_ids'][i])
            cur_ent_ids_pred = ent_ids_outputs['predict'][i_ent: i_ent + num_ent_ids]
            i_ent += num_ent_ids

            ent_ids_dict = {k: v for k, v in zip(cur_ent_ids, cur_ent_ids_pred)}
            cur_rel_pred = []
            cur_bin_rel_pred = []
            tmp_candi_rels = []
            for e1, e2 in cur_candi_rels:
                e1_label = ent_ids_dict[e1].item()
                e2_label = ent_ids_dict[e2].item()
                if self.vocab.get_token_from_index(e1_label, "ent_ids_labels") == "None":
                    i_rel += 1
                    continue
                if self.vocab.get_token_from_index(e2_label, "ent_ids_labels") == "None":
                    i_rel += 1
                    continue
                t_candi_rels = ((e1[0], e1[1]+1), (e2[0], e2[1]+1))
                tmp_candi_rels.append(t_candi_rels)
                cur_rel_pred.append(rel_outputs["predict"][i_rel].item())
                cur_bin_rel_pred.append(bin_rel_outputs["predict"][i_rel].item())
                i_rel += 1
            ret_rel_pred.append(cur_rel_pred)
            ret_candi_rels.append(tmp_candi_rels)
            ret_bin_rel_pred.append(cur_bin_rel_pred)
        return ret_candi_rels, ret_rel_pred, ret_bin_rel_pred

    def create_all_ent_pred(self,
                            batch: Dict[str, Any],
                            ent_ids_outputs: Dict[str, Any]) -> List[List]:
        all_ent_pred = []
        j = 0
        for i in range(len(batch['all_ent_ids'])):
            seq_len = batch['seq_lens'][i]
            cur_ent_pred = [self.vocab.get_token_index("O", "ent_labels")
                            for _ in range(seq_len)]
            cur_ent_ids = batch['all_ent_ids'][i]
            cur_ent_ids_num = len(cur_ent_ids)
            cur_ent_ids_pred = ent_ids_outputs['predict'][j: j + cur_ent_ids_num].cpu().numpy()
            for (start, end), t in zip(cur_ent_ids, cur_ent_ids_pred):
                t_label_text = self.vocab.get_token_from_index(t, "ent_ids_labels")
                if t_label_text == "None":
                    continue
                if start == end:
                    start_label = self.vocab.get_token_index("U-%s" % t_label_text, "ent_labels")
                    cur_ent_pred[start] = start_label
                else:
                    start_label = self.vocab.get_token_index("B-%s" % t_label_text, "ent_labels")
                    end_label = self.vocab.get_token_index("E-%s" % t_label_text, "ent_labels")
                    cur_ent_pred[start] = start_label
                    cur_ent_pred[end] = end_label
                    for k in range(start + 1, end):
                        cur_ent_pred[k] = self.vocab.get_token_index("I-%s" % t_label_text, "ent_labels")
            j += cur_ent_ids_num
            all_ent_pred.append(cur_ent_pred)
        return all_ent_pred


    def generate_all_candidates(
            self,
            batch: Dict[str, Any]) -> (List[List[Tuple[Tuple, Tuple]]], List[List[int]]):
        all_candi_rels = []
        all_rel_labels = []
        k = 0
        for i in range(len(batch['tokens'])):
            gold_candi_rels = batch['candi_rels'][i]
            gold_rel_labels = batch['rel_labels'][i]
            cur_ent_ids = batch['all_ent_ids'][i]

            gold_dict = {((k[0][0], k[0][1]-1), (k[1][0], k[1][1]-1)): v
                         for k, v in zip(gold_candi_rels, gold_rel_labels)}

            cur_candi_rels = []
            cur_rel_labels = []
            for e1 in cur_ent_ids:
                for e2 in cur_ent_ids:
                    if e1[0] >= e2[0]:
                        continue
                    if e1[1] >= e2[0]:
                        continue
                    cur_candi_rels.append((e1, e2))
                    if (e1, e2) in gold_dict:
                        cur_rel_labels.append(gold_dict[(e1, e2)])
                    else:
                        cur_rel_labels.append(self.vocab.get_token_index("None", "rel_labels"))
            all_candi_rels.append(cur_candi_rels)
            all_rel_labels.append(cur_rel_labels)
        return all_candi_rels, all_rel_labels

    def gcn_extractor_test(self, batch, ent_ids_span_feats, rel_batch):
        return ent_ids_span_feats, rel_batch['inputs'] if rel_batch is not None else None

