import random

import torch
import torch.nn as nn
import torch.optim as optim
import json
from copy import deepcopy
from tqdm import tqdm
import nni
from nni.utils import merge_parameter
import torch.nn.functional as F

from new_model import BertEncoder, ProtoSoftmaxLayer, Experience
from data_loader import get_data_loader
from sampler import DataSampler
from transformers import BertTokenizer
from sklearn.metrics import f1_score
from config import get_config
from new_utils import set_seed, select_data, get_proto, eliminate_experiences
import time

import os

from utils import get_aca_data

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"


def evaluate(config, test_data, seen_relations, rel2id, model = None,encoder=None):
    encoder.eval()
    model.eval()
    num = len(test_data)
    data_loader = get_data_loader(config, test_data, batch_size=8)
    gold = []
    pred = []
    correct = 0

    with torch.no_grad():
        seen_relation_ids = [rel2id[rel] for rel in seen_relations]
        for _, (_, labels, sentences, _) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            rep, _ = encoder(sentences)
            logits = model.get_mem_feature(rep)
            predicts = logits.max(dim=-1)[1].cpu()
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            correct += (predicts == labels).sum().item()
            predicts = predicts.tolist()
            labels = labels.tolist()
            gold.extend(labels)
            pred.extend(predicts)

    accuracy = correct / num
    micro_f1 = f1_score(gold, pred, average='micro')
    macro_f1 = f1_score(gold, pred, average='macro')
    print(f"{green_print}Accuracy: {accuracy * 100:.2f}%{default_print}")
    print(f"{green_print}Micro F1 Score: {micro_f1:.4f}{default_print}")
    print(f"{green_print}Macro F1 Score: {macro_f1:.4f}{default_print}")

    return correct / num


def Wake(config, model, sentence_encoder,train_set,experience_pool_1, epochs,rel2id,seen_relations):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    sentence_encoder.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': sentence_encoder.parameters(), 'lr': config.encoder_init_lr},
        {'params': model.parameters(), 'lr': config.classifier_lr}
    ], betas=(0.85, 0.99))
    seen_relation_ids = [rel2id[rel] for rel in seen_relations]
    for epoch_i in range(epochs):
        for step, (_, labels, sentences, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentence_encoder.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            rep,features = sentence_encoder(sentences)
            logits, _= model(rep)
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()

            labels = labels.to(config.device)
            max_label = labels.max().item()
            num_classes = logits.shape[1]
            if max_label >= num_classes:
                raise ValueError(f"标签值 {max_label} 超出模型类别数 {num_classes}")
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(sentence_encoder.parameters(), config.max_grad_norm)
            optimizer.step()
            torch.cuda.empty_cache()
            if epochs - 1 == epoch_i:
                with torch.no_grad():
                    experience_pool_1.add(sentences, labels, logits)

    return model,sentence_encoder,experience_pool_1


def untilize(config, model,sentence_encoder, experiences_pool, train_set,rel2id,seen_relations):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.eval()
    sentence_encoder.eval()
    seen_relation_ids = [rel2id[rel] for rel in seen_relations]

    with torch.no_grad():
        for step, (_, labels, sentences, _) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            rep, _ = sentence_encoder(sentences)
            logits, _ = model(rep)
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            labels = labels.to(config.device)
            experiences_pool.add(sentences, labels, logits)
    return experiences_pool

def utilize_experience(config, experiences_pool, model,sentence_encoder):
    model.train()
    sentence_encoder.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=config.margin_factor)
    optimizer = optim.Adam([
        {'params': sentence_encoder.parameters(), 'lr': config.encoder_lr},
        {'params': model.classifier.parameters(), 'lr': config.classifier_lr}
    ], betas=(0.85, 0.99))

    for _ in range(1):
        for experience in experiences_pool.experiences:
            exp_sentences, labels, exp_logits, _ = experience
            model.zero_grad()
            sentence_encoder.zero_grad()
            rep,_ = sentence_encoder(exp_sentences)
            logits, _ = model(rep)
            labels = labels.cuda()
            ce_loss = criterion(logits, labels)
            soft_log_probs = F.log_softmax(logits / config.temperature, dim=1)
            soft_targets = F.softmax(exp_logits / config.temperature, dim=1)
            KD_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean') * (
                        config.temperature ** 2)
            CE_loss = F.cross_entropy(logits, labels)
            distillation_loss = config.alpha * KD_loss + (1 - config.alpha) * CE_loss
            loss = ce_loss + distillation_loss + lossfn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(sentence_encoder.parameters(), config.max_grad_norm)
            optimizer.step()
            torch.cuda.empty_cache()

    return model,sentence_encoder

def utilize_experience_1(config, experiences_pool, model,sentence_encoder):
    model.train()
    sentence_encoder.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=config.margin_factor)
    optimizer = optim.Adam([
        {'params': sentence_encoder.parameters(), 'lr': config.encoder_lr},
        {'params': model.classifier.parameters(), 'lr': config.classifier_lr}
    ], betas=(0.85, 0.99))

    for _ in range(1):
        high_quality_experiences = experiences_pool.get_high_quality_experiences()
        for experience in high_quality_experiences:
            exp_sentences, labels, exp_logits, _ = experience
            model.zero_grad()
            sentence_encoder.zero_grad()
            rep,_ = sentence_encoder(exp_sentences)
            logits, _ = model(rep)
            labels = labels.cuda()
            ce_loss = criterion(logits, labels)
            soft_log_probs = F.log_softmax(logits / config.temperature, dim=1)
            soft_targets = F.softmax(exp_logits / config.temperature, dim=1)
            KD_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean') * (
                        config.temperature ** 2)
            CE_loss = F.cross_entropy(logits, labels)
            distillation_loss = config.alpha * KD_loss + (1 - config.alpha) * CE_loss
            loss = ce_loss + distillation_loss + lossfn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(sentence_encoder.parameters(), config.max_grad_norm)
            optimizer.step()
            torch.cuda.empty_cache()

    return model,sentence_encoder

def REM(config, model, sentence_encoder,train_set, epochs,rel2id,seen_relations):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    sentence_encoder.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=config.margin_factor)
    optimizer = optim.Adam([
        {'params': sentence_encoder.parameters(), 'lr': config.encoder_lr},
        {'params': model.classifier.parameters(), 'lr': config.classifier_lr}
    ], betas=(0.85, 0.99))
    seen_relation_ids = [rel2id[rel] for rel in seen_relations]
    for epoch_i in range(epochs):
        for step, (_, labels, sentences, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentence_encoder.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            rep,features = sentence_encoder(sentences)
            logits, _ = model(rep)
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            labels = labels.to(config.device)
            ce_loss = criterion(logits, labels)
            loss = ce_loss + lossfn(logits, labels)
            ##########################################
            con_loss = contrastive_loss(config, features, labels, model.prototypes)
            loss += con_loss
            ##########################################
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(sentence_encoder.parameters(), config.max_grad_norm)
            optimizer.step()
            torch.cuda.empty_cache()
    return model,sentence_encoder


def contrastive_loss(config, feature, labels, proto_features=None):
    dot_div_temp = torch.mm(feature, proto_features.T) / config.contrastive_temperature # [batch_size, rel_num]
    dot_div_temp_norm = dot_div_temp - 1.0 / config.contrastive_temperature
    exp_dot_temp = torch.exp(dot_div_temp_norm) + 1e-8 # avoid log(0)

    mask = torch.zeros_like(exp_dot_temp).to(config.device)
    mask.scatter_(1, labels.unsqueeze(1), 1.0)
    cardinalities = torch.sum(mask, dim=1)

    log_prob = -torch.log(exp_dot_temp / torch.sum(exp_dot_temp, dim=1, keepdim=True))
    scloss_per_sample = torch.sum(log_prob*mask, dim=1) / cardinalities
    scloss = torch.mean(scloss_per_sample)
    return scloss



if __name__ == '__main__':
    print("HELLO, NEW_INIT")
    config = get_config()
    tuner_params = nni.get_next_parameter()
    config = merge_parameter(base_params=config, override_params=tuner_params)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path,additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],mean_resizing=False)
    test_cur_record, test_total_record = [], []
    pid2name = json.load(open('data/pid2name.json', 'r')) if config.task_name.lower() == 'fewrel' else {}
    start_time = time.time()
    for rounds in range(config.total_rounds):
        test_cur, test_total = [], []
        set_seed(config.random_seed + rounds * 100)
        sampler = DataSampler(config=config, seed=config.random_seed + rounds * 100, tokenizer=tokenizer)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id


        memorized_samples = {}
        memorized_data = []
        experience_pool = Experience(config)
        experience_pool_1 = Experience(config)
        encoder = BertEncoder(config=config).to(config.device)
        init_model = ProtoSoftmaxLayer(config=config, num_classes=len(sampler.id2rel)).to(config.device)
        model = ProtoSoftmaxLayer(config=config, num_classes=len(sampler.id2rel)).to(config.device)
        for task_id, (training_data, valid_data, test_data, current_relations, historic_test_data,seen_relations) in enumerate(sampler):
            print(f"{yellow_print}Training task {task_id + 1}, relation set {current_relations}.{default_print}")
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            init_model ,encoder,experience_pool_1 = Wake(config, init_model, encoder,train_data_for_initial ,experience_pool_1, config.init_epochs,rel2id,seen_relations)
            print(f'{blue_print}Selecting memory for task {task_id + 1 }...{default_print}')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])
                experience_pool = untilize(config, init_model,encoder,experience_pool , memorized_samples[relation],rel2id,seen_relations)
            mem_data = []
            for rel in memorized_samples:
                mem_data += memorized_samples[rel]
            print(f"{yellow_print}Memory Replay and Activation{default_print}")
            for _ in range(config.reply_epochs):
                protos4train = []
                for relation in seen_relations:
                    proto = get_proto(config, encoder, memorized_samples[relation])
                    proto = proto / proto.norm()
                    protos4train.append(proto)
                protos4train = torch.cat(protos4train, dim=0).detach()
                model.set_memorized_prototypes(protos4train)
                model ,encoder= REM(config, model, encoder,mem_data, 1,rel2id,seen_relations)
            model ,encoder= utilize_experience(config,experience_pool,model,encoder)
            eliminate_experiences(config, experience_pool_1)
            model, encoder = utilize_experience_1(config, experience_pool_1, model, encoder)

            protos4eval = []
            for relation in seen_relations:
                r = init_model.classifier.weight[rel2id[relation]].detach()
                proto = get_proto(config, encoder, memorized_samples[relation], r)
                proto = proto / proto.norm()
                protos4eval.append(proto)
            protos4eval = torch.cat(protos4eval, dim=0).detach()
            init_model.set_memorized_prototypes(protos4eval)
            protos4eval = []
            for relation in seen_relations:
                r = model.classifier.weight[rel2id[relation]].detach()
                proto = get_proto(config, encoder, memorized_samples[relation],r)
                proto = proto / proto.norm()
                protos4eval.append(proto)
            protos4eval = torch.cat(protos4eval, dim=0).detach()
            model.set_memorized_prototypes(protos4eval)
            print(f"{yellow_print}[Evaluation] Task {task_id + 1} in progress{default_print}")
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]
            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]


            cur_acc = evaluate(config, test_data_1, seen_relations, rel2id, model=init_model, encoder=encoder)
            total_acc = evaluate(config, test_data_2, seen_relations, rel2id, model=model,encoder=encoder)
            test_cur.append(cur_acc)
            test_total.append(total_acc)

            print(f"{blue_print}Restart Num: {rounds + 1}{default_print}")
            print(f"{blue_print}Task: {task_id + 1}{default_print}")
            print(f"{green_print}[Evaluation] Current Test Accuracy: {test_cur}{default_print}")
            print(f"{green_print}[Evaluation] History Test Accuracy: {test_total}{default_print}")
            torch.cuda.empty_cache()

        test_cur_record.append(test_cur)
        test_total_record.append(test_total)
        average_history_accuracy = sum(test_total) / len(test_total) if test_total else 0
        print( f"{green_print}[Evaluation] Average History Test Accuracy: {average_history_accuracy:.4f}{default_print}")
        nni.report_intermediate_result(average_history_accuracy)
    print(f"{yellow_print}==========================================================================================final result==========================================================================================================================================={default_print}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    test_cur_record = torch.tensor(test_cur_record)
    test_total_record = torch.tensor(test_total_record)
    test_cur_record = torch.mean(test_cur_record, dim=0).tolist()
    test_total_record = torch.mean(test_total_record, dim=0)
    final_average = torch.mean(test_total_record).item()
    print(f"{green_print}[Evaluation] Current Test Record Mean: {test_cur_record}{default_print}")
    print(f"{green_print}[Evaluation] All Test Records Mean: {test_total_record.tolist()}{default_print}")
    print(f'{green_print}Final average: {final_average}{default_print}')
    print(f"{green_print}运行时间：{minutes} 分 {seconds} 秒{default_print}")
    nni.report_final_result(final_average)
