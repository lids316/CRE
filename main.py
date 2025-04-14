import torch
import torch.nn as nn
import torch.optim as optim
import json
from copy import deepcopy
from tqdm import tqdm
import nni
import torch.nn.functional as F
from nni.utils import merge_parameter

from model import BertEncoder, ProtoSoftmaxLayer, Experience, eliminate_experiences
from data_loader import get_data_loader
from sampler import DataSampler
from transformers import BertTokenizer
from sklearn.metrics import f1_score
from config import get_config
from utils import set_seed, select_data, get_proto, get_aca_data
from loss import AdversarialLoss
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

    # accuracy = correct / num
    # micro_f1 = f1_score(gold, pred, average='micro')
    # macro_f1 = f1_score(gold, pred, average='macro')
    # print(f"{green_print}Accuracy: {accuracy * 100:.2f}%{default_print}")
    # print(f"{green_print}Micro F1 Score: {micro_f1:.4f}{default_print}")
    # print(f"{green_print}Macro F1 Score: {macro_f1:.4f}{default_print}")

    return correct / num


def Wake(config, optimizer, model, experiences_pool, train_set, epochs):
    # 获取数据加载器
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW([
    #     {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
    #     {'params': model.classifier.parameters(), 'lr': 0.001},
    # ], weight_decay=0.0001)
    # optimizer = optim.Adam([
    #     {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
    #     {'params': model.classifier.parameters(), 'lr': 0.001}
    # ], betas=(0.85, 0.99))

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
            if epoch_i == 1:
                experiences_pool.add(sentences, labels, logits)
    return model, experiences_pool


def utilize_experience(config, optimizer, experiences_pool, model):
    model.train()
    criterion = nn.CrossEntropyLoss()
    lossfn = nn.MultiMarginLoss(margin=1)
    # optimizer = optim.Adam([
    #     {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
    #     {'params': model.classifier.parameters(), 'lr': 0.001}
    # ], betas=(0.85, 0.99))

    for _ in range(1):
        high_quality_experiences = experiences_pool.get_high_quality_experiences()
        for experience in high_quality_experiences:
            exp_sentences, labels, exp_logits, _ = experience
            model.zero_grad()
            logits, _ = model(exp_sentences)
            labels = labels.cuda()
            ce_loss = criterion(logits, labels)
            soft_log_probs = F.log_softmax(logits / config.temperature, dim=1)
            soft_targets = F.softmax(exp_logits / config.temperature, dim=1)
            KD_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean') * (
                        config.temperature ** 2)
            CE_loss = F.cross_entropy(logits, labels)
            distillation_loss = config.alpha * KD_loss + (1 - config.alpha) * CE_loss
            loss = ce_loss + distillation_loss
            loss += lossfn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model, experiences_pool


def rem_phase(config, optimizer, model, train_set):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    criterion2 = AdversarialLoss(config)
    lossfn = nn.MultiMarginLoss(margin=1)
    # optimizer = optim.Adam([
    #     {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
    #     {'params': model.classifier.parameters(), 'lr': 0.001}
    # ], betas=(0.85, 0.99))
    for step, (_, labels, sentences, _) in enumerate(tqdm(data_loader)):
        model.zero_grad()
        sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
        logits, _ = model(sentences)
        labels = labels.cuda()
        ce_loss = criterion(logits, labels)
        adv_data = Generate_adversarial(config, sentences)
        adv_logits, _ = model(adv_data)
        conb_loss = criterion2(adv_logits, logits, labels)
        loss = conb_loss + ce_loss
        loss += lossfn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
    return model


def Generate_adversarial(config, x):
    x = x.float()
    noise = torch.randn_like(x)
    norm = torch.norm(noise, p=config.p1, dim=tuple(range(1, x.dim())), keepdim=True)
    noise = noise / norm * config.gamma
    x = x + noise
    return x.long()




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

        add_relation_num = config.relations_per_task * 3
        model = ProtoSoftmaxLayer(config=config,
                                  sentence_encoder=encoder,
                                  num_classes=len(sampler.id2rel) + add_relation_num).to(config.device)

        memorized_samples = {}
        experience_pool = Experience(config)
        optimizer = optim.Adam([
            {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
            {'params': model.classifier.parameters(), 'lr': 0.001}
        ], betas=(0.85, 0.99))

        for task_id, (training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(sampler):
            print(f"{yellow_print}Training task {task_id + 1}, relation set {current_relations}.{default_print}")

            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            add_aca_data = get_aca_data(config, deepcopy(training_data), current_relations, tokenizer)
            model, experience_pool = Wake(config, optimizer, model, experience_pool, train_data_for_initial + add_aca_data, 2)
            eliminate_experiences(config, experience_pool)
            model, experience_pool = utilize_experience(config, optimizer, experience_pool,  model)
            model.incremental_learning(config.num_of_relations, add_relation_num)

            # Step6
            print(f'{blue_print}Selecting memory for task {task_id + 1}...{default_print}')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])

            mem_data = []
            for rel in memorized_samples:
                mem_data += memorized_samples[rel]

            print(f"{yellow_print}Memory Replay and Activation{default_print}")
            for _ in range(2):

                model = rem_phase(config, optimizer, model, mem_data + train_data_for_initial)

            # static_prot = []
            # for relation in current_relations:
            #     proto = get_proto(config, encoder, training_data[relation])
            #     proto = proto / proto.norm()
            #     static_prot.append(proto)
            # static_prot = torch.cat(static_prot, dim=0).detach()
            # model.set_static_prototypes(static_prot)

            protos4eval = []
            for relation in seen_relations:
                r = model.classifier.weight[rel2id[relation]].detach()
                proto = get_proto(config, encoder, memorized_samples[relation], r)
                proto = proto / proto.norm()
                protos4eval.append(proto)
            protos4eval = torch.cat(protos4eval, dim=0).detach()

            model.set_memorized_prototypes(protos4eval)
            # model.adjust_static_prototypes(protos4eval)

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

    print(f"{green_print}[Evaluation] Current Test Record: {test_cur_record}{default_print}")
    print(f"{green_print}[Evaluation] All Test Records: {test_total_record.tolist()}{default_print}")
    print(
        f"{yellow_print}=========================================================================================final mean result==========================================================================================================================================={default_print}")
    test_cur_record = torch.mean(test_cur_record, dim=0).tolist()
    test_total_record = torch.mean(test_total_record, dim=0)
    final_average = torch.mean(test_total_record).item()
    print(f"{green_print}[Evaluation] Current Test Record Mean: {test_cur_record}{default_print}")
    print(f"{green_print}[Evaluation] All Test Records Mean: {test_total_record.tolist()}{default_print}")
    print(f'{green_print}Final average: {final_average}{default_print}')
    print(f"{green_print}运行时间：{minutes} 分 {seconds} 秒{default_print}")
    nni.report_final_result(final_average)
