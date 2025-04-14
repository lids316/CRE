import torch
import torch.nn as nn
import torch.optim as optim
import json
from copy import deepcopy
from tqdm import tqdm
import nni
from nni.utils import merge_parameter

from model import BertEncoder, ProtoSoftmaxLayer
from data_loader import get_data_loader
from sampler import DataSampler
from transformers import BertTokenizer
from sklearn.metrics import f1_score
from config import get_config
from utils import set_seed, select_data, get_proto, get_aca_data
import time

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"


def evaluate(config, test_data, seen_relations, rel2id, model=None):
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
            _, rep = model(sentences)
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


def Wake(config, model, train_set, epochs):
    # 获取数据加载器
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.classifier.parameters(), 'lr': 0.001},
    ], weight_decay=0.0001)

    for epoch_i in range(epochs):
        for step, (_, labels, sentences, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, _ = model(sentences)
            labels = labels.to(config.device)
            max_label = labels.max().item()
            num_classes = logits.shape[1]
            if max_label >= num_classes:
                raise ValueError(f"标签值 {max_label} 超出模型类别数 {num_classes}")
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model


def REM(config, model, train_set, epochs):
    # 获取数据加载器
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=0.0001)

    for epoch_i in range(epochs):
        for step, (_, labels, sentences, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, _ = model(sentences)
            labels = labels.to(config.device)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model


def NREM(config, model, mem_set, epochs, current_proto, seen_relation_ids):
    data_loader = get_data_loader(config, mem_set, shuffle=True)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.sentence_encoder.parameters(), 'lr': config.encoder_lr},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=config.weight_decay, eps=config.adam_epsilon)
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses = []
        for step, (_, labels, sentences, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, rep = model(sentences)
            logits_proto = model.memory_forward(rep)
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            labels = labels.to(config.device)
            loss1 = criterion(logits, labels)
            loss2 = (criterion(logits_proto, labels))
            loss = loss1 + loss2
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model


if __name__ == '__main__':
    config = get_config()
    tuner_params = nni.get_next_parameter()
    config = merge_parameter(base_params=config, override_params=tuner_params)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path,
                                              additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
                                              mean_resizing=False)
    test_cur_record = []
    test_total_record = []
    pid2name = json.load(open('data/pid2name.json', 'r')) if config.task_name.lower() == 'fewrel' else {}
    start_time = time.time()
    for rounds in range(config.total_rounds):
        test_cur = []
        test_total = []
        set_seed(config.random_seed + rounds * 100)
        sampler = DataSampler(config=config, seed=config.random_seed + rounds * 100, tokenizer=tokenizer)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        encoder = BertEncoder(config=config).to(config.device)

        if config.additional_classifier:
            add_relation_num = config.relations_per_task * 3
            model = ProtoSoftmaxLayer(config=config,
                                      sentence_encoder=encoder,
                                      num_classes=len(sampler.id2rel) + add_relation_num).to(config.device)
        else:
            model = ProtoSoftmaxLayer(config=config,
                                      sentence_encoder=encoder,
                                      num_classes=len(sampler.id2rel)
                                      ).to(config.device)

        memorized_samples = {}

        for task_id, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):
            print(f"{yellow_print}Training task {task_id + 1}, relation set {current_relations}.{default_print}")

            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            if config.additional_classifier:
                add_aca_data = get_aca_data(config, deepcopy(training_data), current_relations, tokenizer)

            if config.additional_classifier:
                model = Wake(config, model, train_data_for_initial + add_aca_data, 4)
            else:
                model = Wake(config, model, train_data_for_initial, 2)

            if config.additional_classifier:
                model.incremental_learning(config.num_of_relations, add_relation_num)

            # Step6
            print(f'{blue_print}Selecting memory for task {task_id + 1}...{default_print}')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])

            mem_data = []
            for rel in memorized_samples:
                mem_data += memorized_samples[rel]

            data4step2 = mem_data + train_data_for_initial

            seen_relation_ids = [rel2id[rel] for rel in seen_relations]

            for _ in range(2):
                protos4train = []
                for relation in seen_relations:
                    protos4train.append(get_proto(config, encoder, memorized_samples[relation]))
                protos4train = torch.cat(protos4train, dim=0).detach()
                print(f"{yellow_print}Memory Replay and Activation{default_print}")
                model = REM(config, model, data4step2, 1)
                print(f"{yellow_print}Memory Reconsolidation{default_print}")
                model = NREM(config, model, mem_data, 1, protos4train, seen_relation_ids)

            protos4eval = []
            for relation in seen_relations:
                r = model.classifier.weight[rel2id[relation]].detach()
                proto = get_proto(config, encoder, memorized_samples[relation], r)
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
            cur_acc = evaluate(config, test_data_1, seen_relations, rel2id, model=model)
            total_acc = evaluate(config, test_data_2, seen_relations, rel2id, model=model)

            test_cur.append(cur_acc)
            test_total.append(total_acc)
            print(f"{blue_print}Restart Num: {rounds + 1}{default_print}")
            print(f"{blue_print}Task: {task_id + 1}{default_print}")
            print(f"{green_print}[Evaluation] Current Test Accuracy: {test_cur}{default_print}")
            print(f"{green_print}[Evaluation] History Test Accuracy: {test_total}{default_print}")

        test_cur_record.append(test_cur)
        test_total_record.append(test_total)
        average_history_accuracy = sum(test_total) / len(test_total) if test_total else 0
        print(
            f"{green_print}[Evaluation] Average History Test Accuracy: {average_history_accuracy:.4f}{default_print}")
        nni.report_intermediate_result(average_history_accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(
        f"{yellow_print}==========================================================================================final result==========================================================================================================================================={default_print}")
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
