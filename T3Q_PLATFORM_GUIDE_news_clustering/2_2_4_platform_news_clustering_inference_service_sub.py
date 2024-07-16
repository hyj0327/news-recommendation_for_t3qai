# inference_service_sub.py

import logging
import pandas as pd
import numpy as np
import faiss
import os
from bertopic import BERTopic
import kss
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer

import torch.nn as nn
import math
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertTokenizer, AdamW,  get_linear_schedule_with_warmup
import pytorch_lightning as pl
import json
import requests
from bs4 import BeautifulSoup
import re


from t3qai_client import DownloadFile
import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, \
    T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH


MAX_TOKEN_COUNT = 512
N_EPOCHS = 5
BATCH_SIZE = 4
BERT_MODEL_NAME = 'jinmang2/kpfbert'


def exec_init_model():
    logging.info('[hunmin log] T3QAI_INIT_MODEL_PATH : {}'.format(
        T3QAI_INIT_MODEL_PATH))

    trained_model = Summarizer.load_from_checkpoint(
        os.path.join(T3QAI_INIT_MODEL_PATH, 'best-checkpoint.ckpt'),
        strict=False
    )

    model = BERTopic(
        embedding_model='bongsoo/kpf-sbert-128d-v1', min_topic_size=5)
    embedding_model = SentenceTransformer(
        'bongsoo/kpf-sbert-128d-v1')  # 임베딩 모델

    model_info_dict = {
        "summary_model": trained_model,
        "model": model,
        "embedding_model": embedding_model
    }

    return model_info_dict


def exec_inference_dataframe(df, model_info_dict):

    logging.info(
        '[hunmin log] the start line of the function [exec_inference_dataframe]')

    # 학습 모델 준비
    trained_model = model_info_dict['summary_model']

    # 뉴스 데이터셋 불러오기
    my_path = os.path.join(T3QAI_INIT_MODEL_PATH, 'dataset') + '/'

    news_dataset = pd.read_json(my_path+'news.json')
    summary_embedding_dataset = np.load(my_path+'summary_embedding.npy')
    paragraph_dataset = pd.read_json(my_path+'paragraph_data.json')
    paragraph_embedding_dataset = np.load(my_path+'paragraph_embedding.npy')

    logging.info('[hunmin log] news_dataset : {}'.format(news_dataset.shape))
    logging.info('[hunmin log] summary_embedding_dataset : {}'.format(
        summary_embedding_dataset.shape))
    logging.info('[hunmin log] paragraph_dataset : {}'.format(
        paragraph_dataset.shape))
    logging.info('[hunmin log] paragraph_embedding_dataset : {}'.format(
        paragraph_embedding_dataset.shape))

    # data preprocess
    logging.info('[hunmin log] load dataframe: {}'.format(df))
    target_link = df.iloc[0, 0]
    target_data = fetch_article_data(target_link)
    target_article = target_data['article']

    # Summary
    target_summary_ = summarize_article(trained_model, target_article)
    target_summary = " ".join(target_summary_)
    logging.info('[hunmin log] target_summary : {}'.format(target_summary))

    # Summary -> Embedding
    model = model_info_dict['embedding_model']
    target_summary_embedding = model.encode(
        target_summary, normalize_embeddings=True)

    logging.info('start Similarity')

    # Similarity
    threshold = 0.55
    similar_list = []
    for i in range(len(summary_embedding_dataset)):
        similarity = pearson_similarity(
            target_summary_embedding, summary_embedding_dataset[i])

        if similarity > threshold:
            # threshold 이상이면 유사한 기사 리스트에 추가
            similar_list.append((similarity, i))

    # 유사도 기준 내림차순 정렬
    sorted_similar_list = sorted(
        similar_list, key=lambda x: x[0], reverse=True)

    # 100개 이상이면 100개만 추려서 반환
    if len(similar_list) > 100:
        similar_index_list = [item[1] for item in sorted_similar_list[:100]]

    # 100개 이하면 모두 반환
    else:
        similar_index_list = [item[1] for item in sorted_similar_list]

    logging.info('end Similarity')

    target_paragraphs = split_into_paragraphs(target_article)
    target_paragraph_data = []
    for data in target_paragraphs:
        target_paragraph_data.append([-1]+[data])

    target_paragraph_data = pd.DataFrame(
        data=target_paragraph_data, columns=['index', 'paragraph'])

    paragraph_embedding_dataset = paragraph_embedding_dataset[paragraph_dataset['index'].isin(
        similar_index_list)]
    paragraph_dataset = paragraph_dataset[paragraph_dataset['index'].isin(
        similar_index_list)]

    target_embeddings = model.encode(
        target_paragraph_data['paragraph'].tolist())  # 현재 읽고 있는 기사 단락 임베딩

    train_paragraph_embeddings = np.vstack(
        (target_embeddings, paragraph_embedding_dataset))
    train_paragraph_data = pd.concat(
        [target_paragraph_data, paragraph_dataset], axis=0)

    logging.info('Start BERTopic')

    model = model_info_dict['model']

    topics, probs = model.fit_transform(
        documents=train_paragraph_data['paragraph'], embeddings=train_paragraph_embeddings)  # 클러스터링 만들기
    train_paragraph_data['topic'] = topics
    target_paragraph_data = pd.merge(target_paragraph_data, train_paragraph_data[[
                                     'paragraph', 'topic']], on='paragraph', how='inner')

    # 토픽이 -1, 0은 제외
    target_paragraph_data = target_paragraph_data[target_paragraph_data['topic'] > 0]

    if len(target_paragraph_data) == 0:  # 만약 토픽이 없다면
        print('No Topic')
        various_news_index = similar_index_list  # 유사한 기사 3개 출력

    else:
        paragraph_dataset = pd.merge(paragraph_dataset, train_paragraph_data[[
                                     'paragraph', 'topic']], on='paragraph', how='inner')
        paragraph_dataset = paragraph_dataset[paragraph_dataset['topic'] > 0]

        topic_embeddings = model.topic_embeddings_
        topic_embeddings = topic_embeddings[1:]

        target_topic = target_paragraph_data['topic'].value_counts().idxmax()
        target_topic_embedding = topic_embeddings[target_topic]

        num_topics = len(model.get_topic_freq()) - 1

        # faiss를 이용해서 토픽 간 코사인 유사도 계산
        index = faiss.IndexFlatIP(128)
        faiss.normalize_L2(topic_embeddings)
        index.add(topic_embeddings)
        distances, indices = index.search(np.expand_dims(
            target_topic_embedding, axis=0), num_topics)

        # 가장 유사도가 낮은 토픽 순으로 단락 정렬
        indices = indices[0][::-1]
        indices = np.delete(indices, np.where(indices == 0)[0][0])
        paragraph_dataset['topic'] = pd.Categorical(
            paragraph_dataset['topic'], categories=indices, ordered=True)
        paragraph_dataset = paragraph_dataset.sort_values('topic')

        if num_topics - 2 > 3:
            index_counts = paragraph_dataset.groupby(
                'topic')['index'].value_counts().rename('count').reset_index()
            most_common_index_per_topic = index_counts.loc[index_counts.groupby('topic')[
                'count'].idxmax()]
            most_common_index_per_topic = most_common_index_per_topic.drop_duplicates(
                subset='index')  # 중복 제거

            various_news_index = most_common_index_per_topic['index'].tolist()

        else:  # 토픽이 3개 이하이면 나온 것 모두 반환
            paragraph_dataset = paragraph_dataset.drop_duplicates(
                subset='index')  # 중복 제거
            various_news_index = paragraph_dataset['index'].tolist()

    if target_link in news_dataset['link']:
        same_news_index = news_dataset[news_dataset['link']
                                       == target_link].index
        various_news_index.remove(same_news_index)

    logging.info('END BERTopic')

    various_news = news_dataset.loc[various_news_index][:3]
    result = various_news

    result = {
        "news": {
            "link1": list(various_news[['link']].iloc[0].values),
            "link2": list(various_news[['link']].iloc[1].values),
            "link3": list(various_news[['link']].iloc[2].values)},
        "summary": {
            "sentence1": target_summary_[0],
            "sentence2": target_summary_[1],
            "sentence3": target_summary_[2]}
    }

    logging.info('[hunmin log] result : {}'.format(result))

    return result


###########################################################################
# exec_inference_dataframe() 호출 함수
###########################################################################
class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2):
        super(ExtTransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, hidden_size)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.wo = nn.Linear(hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                    self.linear_keys(query), \
                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                            self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                            layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                        self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            # how can i fix it to use fp16...
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / \
                (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context


class Summarizer(pl.LightningModule):

    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.max_pos = 512
        self.bert = BertModel.from_pretrained(
            BERT_MODEL_NAME)  # , return_dict=True)
        self.ext_layer = ExtTransformerEncoder()
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.loss = nn.BCELoss(reduction='none')

        for p in self.ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    # , input_ids, attention_mask, labels=None):
    def forward(self, src, segs, clss, labels=None):

        mask_src = ~(src == 0)  # 1 - (src == 0)
        mask_cls = ~(clss == -1)  # 1 - (clss == -1)

        top_vec = self.bert(src, token_type_ids=segs, attention_mask=mask_src)
        top_vec = top_vec.last_hidden_state

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)

        loss = 0
        if labels is not None:
            loss = self.loss(sent_scores, labels)

            loss = (loss * mask_cls.float()).sum() / len(labels)

        return loss, sent_scores

    def step(self, batch):

        src = batch['src']
        if len(batch['labels']) > 0:
            labels = batch['labels']
        else:
            labels = None
        segs = batch['segs']
        clss = batch['clss']

        loss, sent_scores = self(src, segs, clss, labels)

        return loss, sent_scores, labels

    def training_step(self, batch, batch_idx):

        loss, sent_scores, labels = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def validation_step(self, batch, batch_idx):

        loss, sent_scores, labels = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def test_step(self, batch, batch_idx):

        loss, sent_scores, labels = self.step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": sent_scores, "labels": labels}

    def acc_loss(self, outputs):
        total_loss = 0
        hit_cnt = 0
        for outp in outputs:
            labels = outp['labels'].cpu()
            predictions, idxs = outp['predictions'].cpu().sort()
            loss = outp['loss'].cpu()
            for label, idx in zip(labels, idxs):
                for i in range(1, 3):
                    if label[idx[-i-1]] == 1:
                        hit_cnt += 1

            total_loss += loss

        avg_loss = total_loss / len(outputs)
        acc = hit_cnt / (3*len(outputs)*len(labels))

        return acc, avg_loss

    def training_epoch_end(self, outputs):

        acc, avg_loss = self.acc_loss(outputs)

        print('acc:', acc, 'avg_loss:', avg_loss)

        self.log('avg_train_loss', avg_loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):

        acc, avg_loss = self.acc_loss(outputs)

        print('val_acc:', acc, 'avg_val_loss:', avg_loss)

        self.log('avg_val_loss', avg_loss, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):

        acc, avg_loss = self.acc_loss(outputs)

        print('test_acc:', acc, 'avg_test_loss:', avg_loss)

        self.log('avg_test_loss', avg_loss, prog_bar=True, logger=True)

        return

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        steps_per_epoch = 11589 // BATCH_SIZE
        total_training_steps = steps_per_epoch * N_EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch,
            num_training_steps=total_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


# 문장 분리 함수
def data_process(text):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    # 문장 분리 하고,
    sents = kss.split_sentences(text)

    # 데이터 가공하고,
    tokenlist = []
    for sent in sents:
        tokenlist.append(tokenizer(
            text=sent,
            add_special_tokens=True))  # , # Add '[CLS]' and '[SEP]'

    src = []  # 토크나이징 된 전체 문단
    labels = []  # 요약문에 해당하면 1, 아니면 0으로 문장수 만큼 생성
    segs = []  # 각 토큰에 대해 홀수번째 문장이면 0, 짝수번째 문장이면 1을 매핑
    clss = []  # [CLS]토큰의 포지션값을 지정

    odd = 0

    for tkns in tokenlist:

        if odd > 1:
            odd = 0
        clss = clss + [len(src)]
        src = src + tkns['input_ids']
        segs = segs + [odd] * len(tkns['input_ids'])
        odd += 1

        # truncation
        if len(src) == MAX_TOKEN_COUNT:
            break
        elif len(src) > MAX_TOKEN_COUNT:
            src = src[:MAX_TOKEN_COUNT - 1] + [src[-1]]
            segs = segs[:MAX_TOKEN_COUNT]
            break

    # padding
    if len(src) < MAX_TOKEN_COUNT:
        src = src + [0]*(MAX_TOKEN_COUNT - len(src))
        segs = segs + [0]*(MAX_TOKEN_COUNT - len(segs))

    if len(clss) < MAX_TOKEN_COUNT:
        clss = clss + [-1]*(MAX_TOKEN_COUNT - len(clss))

    return dict(
        sents=sents,  # 정답 출력을 위해...
        src=torch.tensor(src),
        segs=torch.tensor(segs),
        clss=torch.tensor(clss),
    )

# 요약본 추출 함수


def summarize_test(trained_model, text):
    data = data_process(text.replace('\n', ''))

    # trained_model에 넣어 결과값 반환
    _, rtn = trained_model(data['src'].unsqueeze(
        0), data['segs'].unsqueeze(0), data['clss'].unsqueeze(0))
    rtn = rtn.squeeze()

    # 예측 결과값을 받기 위한 프로세스
    rtn_sort, idx = rtn.sort(descending=True)

    rtn_sort = rtn_sort.tolist()
    idx = idx.tolist()

    end_idx = rtn_sort.index(0)

    rtn_sort = rtn_sort[:end_idx]
    idx = idx[:end_idx]

    if len(idx) > 3:
        rslt = idx[:3]
    else:
        rslt = idx

    summ = []
    for i, r in enumerate(rslt):
        summ.append(data['sents'][r])

    return summ

# 요약본 결과 반환


def summarize_article(trained_model, target_article):
    target_summary = summarize_test(trained_model, target_article)
    return target_summary


# 피어슨 상관계수 구하기
def pearson_similarity(a, b):
    return np.dot((a-np.mean(a)), (b-np.mean(b)))/((np.linalg.norm(a-np.mean(a)))*(np.linalg.norm(b-np.mean(b))))

# 단락 생성


def split_into_paragraphs(article, sentences_per_paragraph=3):
    sentences = kss.split_sentences(article)
    paragraphs = []
    paragraph = []

    for sentence in sentences:
        if len(sentence) > 20:
            # 보통 한 줄에 20자 정도 넘어가야 유의미한 정보가 포함된 문장임
            paragraph.append(sentence)
        if len(paragraph) == sentences_per_paragraph:  # 3줄 이상이면
            paragraphs.append(" ".join(paragraph))  # 3줄을 하나로 합치기
            paragraph = []

        # 남아있는 문장들 중 20자가 넘어가면 단락으로 추가
    if paragraph and len(paragraph) > 20:
        paragraphs.append(" ".join(paragraph))

    return paragraphs  # 단락 데이터 반환


# 크롤링 함수 추가
def preprocessing(d):  # 한국어 기사 본문 전처리 함수
    d = d.lower()
    d = re.sub(r'[a-z0-9\-_.]{3,}@[a-z0-9\-_.]{3,}(?:[.]?[a-z]{2})+', ' ', d)
    d = re.sub(r'‘’ⓒ\'\"“”…=□*◆:/_]', ' ', d)
    d = re.sub(r'\s+', ' ', d)
    d = re.sub(r'^\s|\s$', '', d)
    d = re.sub(r'[<*>_="/■□▷▶]', '', d)
    return d


def fetch_article_data(article_url):  # 기사 본문, 기자 정보 수집 함수
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    resp = requests.get(article_url, headers=headers)
    if resp.status_code != 200:
        return "Failed to retrieve the article"

    article_dom = BeautifulSoup(resp.content, 'html.parser')

    # 특정 선택자를 사용하여 기사 본문 추출
    content_tag = article_dom.select_one(
        'article#dic_area.go_trans._article_content')

    content = preprocessing(content_tag.get_text(
        strip=True)) if content_tag else ''

    # 기자 정보 추출
    reporter_tag = article_dom.select_one('div.byline span') or \
        article_dom.select_one('p.byline') or \
        article_dom.select_one('span.byline')

    reporter = reporter_tag.get_text(strip=True) if reporter_tag else ''

    article_data = {
        "link": article_url,  # 기사 링크
        "article": content,  # 기사 본문
        "reporter": reporter  # 기자
    }

    return article_data
