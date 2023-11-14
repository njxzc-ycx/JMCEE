import json
import numpy as np
import os
import torch
from torch.optim import AdamW
from transformers import BertModel
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random
import math
import logging
import os
import time


def getLogger(name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join("train_log",
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/' + name, mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger

def get_dict(fn):
    with open(fn + '/ty_args.json', 'r', encoding='utf-8') as f:
        ty_args = json.load(f)
    if not os.path.exists(fn + '/shared_args_list.json'):
        args_list = set()
        for ty in ty_args:
            for arg in ty_args[ty]:
                args_list.add(arg)
        args_list = list(args_list)
        with open(fn + '/shared_args_list.json', 'w', encoding='utf-8') as f:
            json.dump(args_list, f, ensure_ascii=False)
    else:
        with open(fn + '/shared_args_list.json', 'r', encoding='utf-8') as f:
            args_list = json.load(f)

    args_s_id = {}
    args_e_id = {}
    for i in range(len(args_list)):
        s = args_list[i] + '_s'
        args_s_id[s] = i
        e = args_list[i] + '_e'
        args_e_id[e] = i

    id_type = {i: item for i, item in enumerate(ty_args)}
    type_id = {item: i for i, item in enumerate(ty_args)}

    id_args = {i: item for i, item in enumerate(args_list)}
    args_id = {item: i for i, item in enumerate(args_list)}
    ty_args_id = {}
    for ty in ty_args:
        args = ty_args[ty]
        tmp = [args_id[a] for a in args]
        ty_args_id[type_id[ty]] = tmp
    return type_id, id_type, args_id, id_args, ty_args, ty_args_id, args_s_id, args_e_id


def read_labeled_data(fn):
    ''' Read Train Data'''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    data_type = []
    data_occur = []
    data_triggers = []
    data_index = []
    data_args = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict.get('id', 0))
        data_occur.append(line_dict['occur'])
        data_type.append(line_dict['type'])
        data_content.append(line_dict['content'])
        data_index.append(line_dict['index'])
        data_triggers.append(line_dict['triggers'])
        data_args.append(line_dict['args'])
    return data_ids, data_occur, data_type, data_content, data_triggers, data_index, data_args


def read_unlabeled_data(fn):
    ''' Read Dev/Test Data'''
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_ids = []
    data_content = []
    for line in lines:
        line_dict = json.loads(line.strip())
        data_ids.append(line_dict['id'])
        data_content.append(line_dict['content'])
    return data_ids, data_content

def get_trigger_mask(start_idx, end_idx, length):
    '''
    used to generate trigger mask, where the element of start/end postion is 1
    [000010100000]
    '''
    mask = np.zeros(length)
    mask[start_idx: end_idx + 1] = 1
    return mask

class MyDataset(Dataset):
    def __init__(self, task, data_ids, data_content, type_id, id_type, args_id, id_args, ty_args,
                 ty_args_id, args_s_id, args_e_id, max_len, device, tokenizer, data_occur=None,
                 data_type=None, data_triggers=None, data_index=None, data_args=None):
        assert task in ['train', 'dev', 'test']
        self.task = task
        self.data_ids = data_ids
        self.data_content = data_content
        self.data_occur = data_occur
        self.data_type = data_type
        self.data_triggers = data_triggers
        self.data_index = data_index
        self.data_args = data_args
        self.type_id = type_id
        self.type_num = len(type_id.keys())
        self.id_type = id_type
        self.args_id = args_id
        self.id_args = id_args
        self.ty_args = ty_args
        self.ty_args_id = ty_args_id
        self.args_s_id = args_s_id
        self.args_num = len(args_s_id.keys())
        self.args_e_id = args_e_id
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.task == "train":
            id = self.data_ids[idx]
            content = self.data_content[idx]
            occur = self.data_occur[idx]
            type = self.data_type[idx]
            triggers = self.data_triggers[idx]
            index = self.data_index[idx]
            args = self.data_args[idx]

            return id, content, occur, type, triggers, index, args
        else:
            id = self.data_ids[idx]
            content = self.data_content[idx]
            return id, content

    def __len__(self):
        return self.data_ids.__len__()

    def data_to_id(self, data_content):

        # default uncased
        data_content = [token.lower() for token in data_content]
        data_content = list(data_content)
        # Here we add <CLS> and <SEP> token for BERT input
        inputs = self.tokenizer.encode_plus(data_content, add_special_tokens=True, max_length=self.max_len, truncation=True, padding='max_length')
        token_id, seg_id, mask_id = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']

        return token_id, seg_id, mask_id

    def type_to_id(self, data_type, data_occur):

        data_type_id = self.type_id[data_type]
        type_vec = np.array([0] * self.type_num)
        for occ in data_occur:
            idx = self.type_id[occ]
            type_vec[idx] = 1

        return data_type_id, type_vec

    def get_rp_tm(self, trigger, index):
        '''
        get relative position embedding and trigger mask, according to the trigger span
        r_pos: relation position embedding
        t_m: trigger mask, used for mean pooling
        '''

        span = trigger[index]
        # plus 1 for additional <CLS> token
        tri_mask = get_trigger_mask(span[0] + 1, span[1] + 1 - 1, self.max_len)

        return tri_mask


    def trigger_seq_id(self, triggers):
        '''
        given trigger span, return ground truth trigger matrix, for bce loss
        t_s: trigger start sequence, 1 for position 0
        t_e: trigger end sequence, 1 for position 0
        '''
        tri_s = np.zeros(shape=[self.type_num, self.max_len])
        tri_e = np.zeros(shape=[self.type_num, self.max_len])

        # for t in trigger:
        #     # plus 1 for additional <CLS> token
        #     tri_s[type][t[0] + 1] = 1
        #     tri_e[type][t[1] + 1 - 1] = 1

        for type_name in triggers:
            type_i = self.type_id[type_name]

            for trigger in triggers[type_name]:
                # plus 1 for additional <CLS> token
                tri_s[type_i][trigger[0] + 1] = 1
                tri_e[type_i][trigger[1] + 1 - 1] = 1

        return tri_s, tri_e

    def args_seq_id(self, args):
        '''
        given argument span, return ground truth argument matrix, for bce loss
        '''
        arg_s = np.zeros(shape=[self.args_num, self.max_len])
        arg_e = np.zeros(shape=[self.args_num, self.max_len])
        arg_mask = [0] * self.args_num
        for args_name in args:
            s_r_i = self.args_s_id[args_name + '_s']
            e_r_i = self.args_e_id[args_name + '_e']
            arg_mask[s_r_i] = 1
            for span in args[args_name]:
                # plus 1 for additional <CLS> token
                arg_s[s_r_i][span[0] + 1] = 1
                arg_e[e_r_i][span[1] + 1 - 1] = 1

        return arg_s, arg_e, arg_mask

    def collate_fn(self, batch):
        if self.task == "train":
            batch_data_id = []
            batch_type_id = []
            batch_type_vec = []
            batch_token_ids = []
            batch_seg = []
            batch_mask = []
            batch_tri_index = []
            batch_tri_mask = []
            batch_tri_s = []
            batch_tri_e = []
            batch_arg_s = []
            batch_arg_e = []
            batch_arg_mask = []
            for id, content, occur, type, triggers, index, args in batch:
                tokens_ids, segs_ids, masks_ids = self.data_to_id(content)
                type_id, type_vec = self.type_to_id(type, occur)
                tri_mask = self.get_rp_tm(triggers[type], index)
                tri_s, tri_e = self.trigger_seq_id(triggers)
                arg_s, arg_e, arg_mask = self.args_seq_id(args)
                batch_data_id.append(id)
                batch_type_id.append(type_id)
                batch_type_vec.append(type_vec)
                batch_token_ids.append(tokens_ids)
                batch_seg.append(segs_ids)
                batch_mask.append(masks_ids)
                batch_tri_index.append(index)
                batch_tri_mask.append(tri_mask)
                batch_tri_s.append(tri_s)
                batch_tri_e.append(tri_e)
                batch_arg_s.append(arg_s)
                batch_arg_e.append(arg_e)
                batch_arg_mask.append(arg_mask)
            batch_type_id = torch.tensor(batch_type_id, dtype=torch.long, device=self.device)
            batch_type_vec = torch.tensor(batch_type_vec, dtype=torch.float, device=self.device)
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
            batch_seg = torch.tensor(batch_seg, dtype=torch.long, device=self.device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.long, device=self.device)
            batch_tri_index = torch.tensor(batch_tri_index, dtype=torch.long, device=self.device)
            batch_tri_mask = torch.tensor(batch_tri_mask, dtype=torch.long, device=self.device)
            batch_tri_s = torch.tensor(batch_tri_s, dtype=torch.float, device=self.device)
            batch_tri_e = torch.tensor(batch_tri_e, dtype=torch.float, device=self.device)
            batch_arg_s = torch.tensor(batch_arg_s, dtype=torch.float, device=self.device)
            batch_arg_e = torch.tensor(batch_arg_e, dtype=torch.float, device=self.device)
            batch_arg_mask = torch.tensor(batch_arg_mask, dtype=torch.long, device=self.device)
            return batch_data_id, batch_type_id, batch_type_vec, batch_token_ids, batch_seg, batch_mask, batch_tri_index, batch_tri_mask, batch_tri_s, batch_tri_e, batch_arg_s, batch_arg_e, batch_arg_mask
        else:
            batch_data_id = []
            batch_content = []
            batch_token_ids = []
            batch_seg = []
            batch_mask = []
            for id, content in batch:
                tokens_ids, segs_ids, masks_ids = self.data_to_id(content)
                batch_data_id.append(id)
                batch_content.append(content)
                batch_token_ids.append(tokens_ids)
                batch_seg.append(segs_ids)
                batch_mask.append(masks_ids)
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
            batch_seg = torch.tensor(batch_seg, dtype=torch.long, device=self.device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.long, device=self.device)
            return batch_data_id, batch_content, batch_token_ids, batch_seg, batch_mask

class TriggerRec(nn.Module):
    def __init__(self, num_types, hidden_size):
        super(TriggerRec, self).__init__()
        self.head_cls = nn.Linear(hidden_size, num_types, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_types, bias=True)


    def forward(self,text_emb):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''
        p_s = torch.sigmoid(self.head_cls(text_emb))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(text_emb))  # [b, t, 1]
        return p_s, p_e

class ArgsRec(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ArgsRec, self).__init__()

        self.head_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_labels, bias=True)


    def forward(self, text_emb, trigger_mask):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t, e]
        :param relative_pos: [b, t, e]
        :param trigger_mask: [b, t]
        :param mask:
        :param type_emb: [b, e]
        :return:  [b, t, a], []
        '''
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2

        inp = torch.add(trigger_emb.unsqueeze(1), text_emb)

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l]
        p_e = torch.sigmoid(self.tail_cls(inp))

        return p_s, p_e

class JMCEE(nn.Module):
    def __init__(self, model_weight, args_num, type_num, hidden_size, max_len):
        super(JMCEE, self).__init__()
        self.bert = model_weight
        self.max_len = max_len
        self.args_num = args_num
        self.type_num = type_num
        self.hidden_size = hidden_size

        self.trigger_rec = TriggerRec(type_num, hidden_size)
        self.args_rec = ArgsRec(hidden_size, self.args_num)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

    def forward(self, tokens, segment, mask, trigger_s_vec, trigger_e_vec, trigger_mask, args_s_vec, args_e_vec, args_mask):
        '''

        :param tokens: [b, t]
        :param segment: [b, t]
        :param mask: [b, t], 0 if masked
        :param trigger_s: [b, t]
        :param trigger_e: [b, t]
        :param relative_pos:
        :param trigger_mask: [0000011000000]
        :param args_s: [b, l, t]
        :param args_e: [b, l, t]
        :param args_m: [b, k]
        :return:
        '''

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        output_emb = outputs[0]


        p_s, p_e = self.trigger_rec(output_emb)

        trigger_loss_s = self.loss_1(p_s, trigger_s_vec.transpose(1, 2))
        trigger_loss_e = self.loss_1(p_e, trigger_e_vec.transpose(1, 2))
        mask_t = mask.unsqueeze(-1).expand_as(trigger_loss_s).float()
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))

        p_s, p_e = self.args_rec(output_emb, trigger_mask)

        args_loss_s = self.loss_2(p_s, args_s_vec.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.loss_2(p_e, args_e_vec.transpose(1, 2))
        mask_a = mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = torch.sum(args_loss_s.mul(mask_a))
        args_loss_e = torch.sum(args_loss_e.mul(mask_a))

        trigger_loss = trigger_loss_s + trigger_loss_e
        args_loss = args_loss_s + args_loss_e

        trigger_loss = 1 * trigger_loss
        args_loss = 1 * args_loss
        loss = trigger_loss + args_loss
        return loss, trigger_loss, args_loss

    def plm(self, tokens, segment, mask):
        assert tokens.size(0) == 1

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]
        return output_emb

    def predict_trigger(self, text_emb, mask):
        assert text_emb.size(0) == 1
        p_s, p_e = self.trigger_rec(text_emb)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.max_len, self.type_num).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.max_len, self.type_num).data.cpu().numpy()
        return p_s, p_e

    def predict_args(self, text_emb, trigger_mask, mask):
        assert text_emb.size(0) == 1
        p_s, p_e= self.args_rec(text_emb, trigger_mask)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.max_len, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.max_len, self.args_num).data.cpu().numpy()
        return p_s, p_e

def gen_idx_event_dict(records):
    data_dict = {}
    for line in records:
        idx = line['id']
        events = line['events']
        data_dict[idx] = events
    return data_dict

def read_jsonl(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data

def gen_tuples(record):
    if record:
        ti, tc, ai, ac = [], [], [], []
        for event in record:
            typ, trigger_span = event['type'], event['trigger']['span']
            ti_one = (trigger_span[0], trigger_span[1])
            tc_one = (typ, trigger_span[0], trigger_span[1])
            ti.append(ti_one)
            tc.append(tc_one)
            for arg_role in event['args']:
                for arg_role_one in event['args'][arg_role]:
                    ai_one = (typ, arg_role_one['span'][0], arg_role_one['span'][1])
                    ac_one = (typ, arg_role_one['span'][0], arg_role_one['span'][1], arg_role)

                    ai.append(ai_one)
                    ac.append(ac_one)
        return ti, tc, ai, ac
    else:
        return [], [], [], []

def score(preds_tuple, golds_tuple):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n = 0, 0, 0
    for sentence_id in golds_tuple:
        gold_sentence_mentions = golds_tuple[sentence_id]
        pred_sentence_mentions = preds_tuple[sentence_id]
        gold_sentence_mentions = set(gold_sentence_mentions)
        pred_sentence_mentions = set(pred_sentence_mentions)
        for mention in pred_sentence_mentions:
            pred_mention_n += 1
        for mention in gold_sentence_mentions:
            gold_mention_n += 1
        for mention in pred_sentence_mentions:
            if mention in gold_sentence_mentions:
                true_positive_n += 1
    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_mention_n != 0:
        prec_c = true_positive_n / pred_mention_n
    else:
        prec_c = 0
    if gold_mention_n != 0:
        recall_c = true_positive_n / gold_mention_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0
    return prec_c, recall_c, f1_c

def cal_scores_ti_tc_ai_ac(preds, golds):
    '''
    :param preds: {id: [{type:'', 'trigger':{'span':[], 'word':[]}, args:[role1:[], role2:[], ...}, ...]}
    :param golds:
    :return:
    '''
    # assert len(preds) == len(golds)
    tuples_pred = [{}, {}, {}, {}]  # ti, tc, ai, ac
    tuples_gold = [{}, {}, {}, {}]  # ti, tc, ai, ac

    for idx in golds:
        if idx not in preds:
            pred = None
        else:
            pred = preds[idx]
        gold = golds[idx]

        ti, tc, ai, ac = gen_tuples(pred)
        tuples_pred[0][idx] = ti
        tuples_pred[1][idx] = tc
        tuples_pred[2][idx] = ai
        tuples_pred[3][idx] = ac

        ti, tc, ai, ac = gen_tuples(gold)
        tuples_gold[0][idx] = ti
        tuples_gold[1][idx] = tc
        tuples_gold[2][idx] = ai
        tuples_gold[3][idx] = ac

    prf_s = []
    for i in range(4):
        prf = score(tuples_pred[i], tuples_gold[i])
        prf_s.append(prf)
    return prf_s

def write_jsonl(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')

if __name__ == "__main__":
    type_id, id_type, args_id, id_args, ty_args, ty_args_id, args_s_id, args_e_id = get_dict("./data/DuEE/proed_data")
    train_data_ids, train_data_occur, train_data_type, train_data_content, train_data_triggers, train_data_index, train_data_args = read_labeled_data(
        "./data/DuEE/proed_data/pro_train.json")
    dev_data_ids, dev_data_content = read_unlabeled_data("./data/DuEE/new_data/new_dev.json")
    test_data_ids, test_data_content = read_unlabeled_data("./data/DuEE/new_data/new_test.json")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    random_seed = 2022126
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = BertTokenizer.from_pretrained('./chinese-roberta-wwm-ext-pytorch')
    type_num = len(type_id.keys())
    args_num = len(args_s_id.keys())
    max_len = 400
    train_batchsize = 2
    dev_batchsize = 1
    epoch = 20
    decoder_num_head = 1
    pos_emb_size = 64
    decoder_dropout = 0.3
    hidden_size = 768
    lr_bert = 2e-5
    lr_task = 1e-4
    logger = getLogger(name="JMCEE_DuEE.txt")
    train_datasets = MyDataset(task="train",
                               data_ids=train_data_ids,
                               data_content=train_data_content,
                               type_id=type_id,
                               id_type=id_type,
                               args_id=args_id,
                               id_args=id_args,
                               ty_args=ty_args,
                               ty_args_id=ty_args_id,
                               args_s_id=args_s_id,
                               args_e_id=args_e_id,
                               max_len=max_len,
                               device=device,
                               tokenizer=tokenizer,
                               data_occur=train_data_occur,
                               data_type=train_data_type,
                               data_triggers=train_data_triggers,
                               data_index=train_data_index,
                               data_args=train_data_args)
    train_dataloader = DataLoader(train_datasets, batch_size=train_batchsize, shuffle=True, collate_fn=train_datasets.collate_fn)

    dev_datasets = MyDataset(task="dev",
                               data_ids=dev_data_ids,
                               data_content=dev_data_content,
                               type_id=type_id,
                               id_type=id_type,
                               args_id=args_id,
                               id_args=id_args,
                               ty_args=ty_args,
                               ty_args_id=ty_args_id,
                               args_s_id=args_s_id,
                               args_e_id=args_e_id,
                               max_len=max_len,
                               device=device,
                               tokenizer=tokenizer)
    dev_dataloader = DataLoader(dev_datasets, batch_size=dev_batchsize, shuffle=True, collate_fn=dev_datasets.collate_fn)

    model_weight = BertModel.from_pretrained("./chinese-roberta-wwm-ext-pytorch",
                                               from_tf=bool('.ckpt' in "./chinese-roberta-wwm-ext-pytorch"),
                                               cache_dir="./plm" if "./plm" else None)
    model = JMCEE(model_weight, args_num, type_num, hidden_size, max_len)
    model = model.to(device)
    bert_params = list(map(id, model.bert.parameters()))

    other_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    optimizer_grouped_parameters = [{'params': model.bert.parameters()}, {'params': other_params, 'lr': lr_task}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr_bert)

    total_loss = 0.0
    te_loss = 0.0
    ae_loss = 0.0
    best_f1 = 0.0
    best_epoch = 0
    for epo in range(epoch):
        model.train()
        step = 0
        for batch_data_id, batch_type_id, batch_type_vec, batch_token_ids, batch_seg, batch_mask, batch_tri_index, \
            batch_tri_mask, batch_tri_s, batch_tri_e, batch_arg_s, batch_arg_e, batch_arg_mask in tqdm(train_dataloader):
            loss, trigger_loss, args_loss = model(batch_token_ids, batch_seg, batch_mask,
                                                             batch_tri_s, batch_tri_e, batch_tri_mask,
                                                             batch_arg_s, batch_arg_e, batch_arg_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            total_loss += loss.item()
            te_loss += trigger_loss.item()
            ae_loss += args_loss.item()
            if step % 5 == 0:
                logger.info(
                    "Epoch id: {}, Training steps: {},TE loss:{:.6f}, AE loss:{:.6f},  Avg loss: {:.6f}".format(
                        epo, step, te_loss / 5, ae_loss / 5,
                        total_loss / 5))
                total_loss = 0.0
                ed_loss = 0.0
                te_loss = 0.0
                ae_loss = 0.0
        model.eval()
        results = []
        for batch_data_id, batch_content, batch_token_ids, batch_seg, batch_mask in tqdm(dev_dataloader):
            idx = batch_data_id[0]
            content = batch_content[0]
            result = {'id': idx, 'content': content}
            text_emb = model.plm(batch_token_ids, batch_seg, batch_mask)
            p_ts, p_te = model.predict_trigger(text_emb, batch_mask)
            events_pred = []
            p_ts = np.transpose(p_ts)
            p_te = np.transpose(p_te)

            for i in range(len(id_type)):
                trigger_s = np.where(p_ts[i] > 0.5)[0]
                trigger_e = np.where(p_te[i] > 0.5)[0]
                trigger_spans = []

                for t in trigger_s:
                    es = trigger_e[trigger_e >= t]
                    if len(es) > 0:
                        e = es[0]
                        if e - t + 1 <= 5:
                            trigger_spans.append((t, e))
                if trigger_spans:
                    for k, span in enumerate(trigger_spans):
                        tm = get_trigger_mask(span[0], span[1], max_len)
                        tm = torch.LongTensor(tm).to(device)
                        tm = tm.unsqueeze(0)

                        p_as, p_ae = model.predict_args(text_emb, tm, batch_mask)
                        p_as = np.transpose(p_as)
                        p_ae = np.transpose(p_ae)

                        type_name = id_type[i]
                        pred_event_one = {'type': type_name}
                        pred_trigger = {'span': [int(span[0]) - 1, int(span[1]) + 1 - 1],
                                        'word': content[int(span[0]) - 1:int(span[1]) + 1 - 1]}  # remove <CLS> token
                        pred_event_one['trigger'] = pred_trigger
                        pred_args = {}
                        args_candidates = ty_args_id[i]
                        for a in args_candidates:
                            pred_args[id_args[a]] = []
                            args_s = np.where(p_as[a] > 0.5)[0]
                            args_e = np.where(p_ae[a] > 0.5)[0]
                            for j in args_s:
                                es = args_e[args_e >= j]
                                if len(es) > 0:
                                    e = es[0]
                                    # if e - j + 1 <= args_len_dict[i]:
                                    pred_arg = {'span': [int(j) - 1, int(e) + 1 - 1],
                                                'word': content[int(j) - 1:int(e) + 1 - 1]}  # remove <CLS> token
                                    pred_args[id_args[a]].append(pred_arg)
                        pred_event_one['args'] = pred_args
                        events_pred.append(pred_event_one)
            result['events'] = events_pred
            results.append(result)
        pred_records = results
        pred_dict = gen_idx_event_dict(pred_records)
        gold_records = read_jsonl("./data/DuEE/new_data/new_dev.json")
        gold_dict = gen_idx_event_dict(gold_records)
        prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
        metric_names = ['TI', 'TC', 'AI', 'AC']
        f1_mean_all = 0.
        for i, prf in enumerate(prf_s):
            f1_mean_all += prf[2]
            logger.info('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[i], prf[0] * 100, prf[1] * 100,
                                                                  prf[2] * 100))
        # write_jsonl(pred_records, f"./{epo}_dev.json")
        f1_mean_all = f1_mean_all / 4
        if f1_mean_all > best_f1:
            best_f1 = f1_mean_all
            logger.info("保存模型中！")
            torch.save(model.state_dict(), "best_model_JMCEE_DuEE.pth")

    best_model = JMCEE(model_weight, args_num, type_num, hidden_size, max_len)
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load('./best_model_JMCEE_DuEE.pth'))
    best_model.eval()
    results = []
    for batch_data_id, batch_content, batch_token_ids, batch_seg, batch_mask in tqdm(dev_dataloader):
        idx = batch_data_id[0]
        content = batch_content[0]
        result = {'id': idx, 'content': content}
        text_emb = model.plm(batch_token_ids, batch_seg, batch_mask)
        p_ts, p_te = model.predict_trigger(text_emb, batch_mask)
        events_pred = []
        p_ts = np.transpose(p_ts)
        p_te = np.transpose(p_te)

        for i in range(len(id_type)):
            trigger_s = np.where(p_ts[i] > 0.5)[0]
            trigger_e = np.where(p_te[i] > 0.5)[0]
            trigger_spans = []

            for t in trigger_s:
                es = trigger_e[trigger_e >= t]
                if len(es) > 0:
                    e = es[0]
                    if e - t + 1 <= 5:
                        trigger_spans.append((t, e))
            if trigger_spans:
                for k, span in enumerate(trigger_spans):
                    tm = get_trigger_mask(span[0], span[1], max_len)
                    tm = torch.LongTensor(tm).to(device)
                    tm = tm.unsqueeze(0)

                    p_as, p_ae = model.predict_args(text_emb, tm, batch_mask)
                    p_as = np.transpose(p_as)
                    p_ae = np.transpose(p_ae)

                    type_name = id_type[i]
                    pred_event_one = {'type': type_name}
                    pred_trigger = {'span': [int(span[0]) - 1, int(span[1]) + 1 - 1],
                                    'word': content[int(span[0]) - 1:int(span[1]) + 1 - 1]}  # remove <CLS> token
                    pred_event_one['trigger'] = pred_trigger
                    pred_args = {}
                    args_candidates = ty_args_id[i]
                    for a in args_candidates:
                        pred_args[id_args[a]] = []
                        args_s = np.where(p_as[a] > 0.5)[0]
                        args_e = np.where(p_ae[a] > 0.5)[0]
                        for j in args_s:
                            es = args_e[args_e >= j]
                            if len(es) > 0:
                                e = es[0]
                                # if e - j + 1 <= args_len_dict[i]:
                                pred_arg = {'span': [int(j) - 1, int(e) + 1 - 1],
                                            'word': content[int(j) - 1:int(e) + 1 - 1]}  # remove <CLS> token
                                pred_args[id_args[a]].append(pred_arg)
                    pred_event_one['args'] = pred_args
                    events_pred.append(pred_event_one)
        result['events'] = events_pred
        results.append(result)
    pred_records = results
    pred_dict = gen_idx_event_dict(pred_records)
    gold_records = read_jsonl("./data/DuEE/new_data/new_dev.json")
    gold_dict = gen_idx_event_dict(gold_records)
    prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
    metric_names = ['TI', 'TC', 'AI', 'AC']
    f1_mean_all = 0.
    for i, prf in enumerate(prf_s):
        f1_mean_all += prf[2]
        logger.info('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[i], prf[0] * 100, prf[1] * 100,
                                                        prf[2] * 100))
    write_jsonl(pred_records, f"./test_result_JMCEE_DuEE.json")