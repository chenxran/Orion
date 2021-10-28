import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification, BertModel)

if not os.path.exists('logs/'):
    os.mkdir('logs/')

logging.basicConfig(
    filename='logs/expbert-{}.log'.format(str(datetime.now())),
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


TASK2PATH = {
    "disease-train": "data/disease/train.txt",
    "disease-test": "data/disease/test.txt",
    "spouse-train": "data/spouse/train.txt",
    "spouse-test": "data/spouse/test.txt",
}

ANNOTATED_EXP = {
    "spouse": "data/exp/expbert_spouse_explanation.txt",
    "disease": "data/exp/expbert_disease_explanation.txt",
}

GENERATED_EXP = {
    "spouse": "data/exp/orion_spouse_explanation.txt",
    "disease": "data/exp/orion_disease_explanation.txt",
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_config(config):
    config = vars(config)
    logger.info("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (25 - len(key)))
        logger.info("{} -->   {}".format(keystr, val))
    logger.info("**************** MODEL CONFIGURATION ****************")


class ExpBERT(nn.Module):
    def __init__(self, args, exp_num):
        super(ExpBERT, self).__init__()
        self.args = args
        self.exp_num = exp_num
        self.config = AutoConfig.from_pretrained(args.model)
        self.model = AutoModel.from_pretrained(args.model, config=self.config)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.config.hidden_size * exp_num, 2)

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, inputs):
        for k, v in inputs["encoding"].items():
            inputs["encoding"][k] = v.cuda()
        pooler_output = self.model(**inputs["encoding"]).last_hidden_state[:, 0, :].reshape(1, self.exp_num * self.config.hidden_size)
        pooler_output = self.dropout(pooler_output)
        logits = self.linear(pooler_output)

        loss = self.criterion(logits, torch.LongTensor([inputs["label"]]).cuda())
        prediction = torch.argmax(logits) 

        return {
            "loss": loss,
            "prediction": prediction,
        }


class REDataset(Dataset):
    def __init__(self, path, exp, tokenizer):
        super(REDataset, self).__init__()
        self.tokenizer = tokenizer
        self.exp = exp
        self.sentences = []
        self.labels = []
        self.entities = []
        with open(path, "r", encoding="utf-8") as file:
            data = file.readlines()
            for example in data:
                sentence, entity1, entity2, id, label = example.strip().split("\t")
                self.sentences.append(sentence)
                if eval(label) == 1:
                    self.labels.append(1)
                elif eval(label) == -1:
                    self.labels.append(0)

                self.entities.append([entity1, entity2])

        logger.info("Number of Example in {}: {}".format(path, str(len(self.labels))))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return {
            "sentence": self.sentences[index],
            "entity": self.entities[index],
            "label": self.labels[index],
        }
    
    def collate_fn(self, batch):
        outputs = []
        for ex in batch:
            temp = []
            for exp in self.exp:
                if "{e1}" in exp or "{e2}" in exp:
                    exp = exp.replace("{e1}", ex["entity"][0]).replace("{e2}", ex["entity"][1])
                else:
                    for entity in ex["entity"]:
                        index = exp.index('<mask>')
                        exp = exp[:index] + entity + exp[index + len('<mask>'):]
                temp.append(exp)
            outputs.append(
                {
                    "encoding": self.tokenizer(
                                    [ex["sentence"]] * len(temp), temp,
                                    add_special_tokens=True,
                                    padding="longest",
                                    truncation=True,
                                    max_length=156,
                                    return_tensors="pt",
                                    return_attention_mask=True,
                                    return_token_type_ids=True,
                                ),
                    "label": ex["label"],
                }
            )
        return outputs

    def collate_fn_(self, batch):
        texts = []
        labels = []
        for ex in batch:
            texts.append(ex["sentence"])
            labels.append(ex["label"])
        
        outputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=156,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        outputs["labels"] = torch.LongTensor(labels)

        return outputs


class Trainer(object):
    def __init__(self, args):
        self.args = args
        print_config(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        
        TASK2EXP = GENERATED_EXP if args.generated_rules else ANNOTATED_EXP
        with open(TASK2EXP[args.task], "r", encoding="utf-8") as file:
            exp = file.readlines()

        self.train_dataset = REDataset(TASK2PATH['{}-train'.format(args.task)], exp, self.tokenizer)
        self.test_dataset = REDataset(TASK2PATH['{}-test'.format(args.task)], exp, self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model).cuda() if self.args.no_exp else ExpBERT(args, len(exp)).cuda()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            collate_fn=self.train_dataset.collate_fn_ if self.args.no_exp else self.train_dataset.collate_fn,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            collate_fn=self.test_dataset.collate_fn_ if self.args.no_exp else self.test_dataset.collate_fn,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
    
    def compute_metrics(self, labels, predictions):
        accuracy = accuracy_score(y_pred=predictions, y_true=labels)
        f1 = f1_score(y_pred=predictions, y_true=labels)

        return accuracy, f1

    def train(self):
        self.model.train()
        self.test(-1)
        for e in range(self.args.epochs):
            with tqdm(total=len(self.train_loader)) as pbar:
                for step, examples in enumerate(self.train_loader):
                    self.model.zero_grad()
                    if self.args.no_exp:
                        for k, v in examples.items():
                            examples[k] = v.cuda()
                        outputs = self.model(**examples)
                        outputs.loss.backward()

                    else:
                        for ex in examples:
                            outputs = self.model(ex)
                            (outputs["loss"] / len(examples)).backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    pbar.update(1)

            self.test(e)

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as pbar:
                loss = []
                labels = []
                predictions = []
                for step, examples in enumerate(self.test_loader):
                    if self.args.no_exp:
                        for k, v in examples.items():
                            examples[k] = v.cuda()
                        outputs = self.model(**examples)
                        loss.append(outputs.loss.float())
                        labels.extend(examples["labels"].tolist())
                        predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())
                    
                    else:
                        for ex in examples:
                            labels.append(ex['label'])
                            outputs = self.model(ex)
                            loss.append(outputs["loss"].item())
                            predictions.append(outputs['prediction'].tolist())

                    pbar.update(1)
                accuracy, f1 = self.compute_metrics(predictions, labels)
            logger.info("[EPOCH {}] Accuracy: {} | F1-Score: {}. (Number of Data {})".format(epoch, accuracy, f1, len(predictions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="spouse")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--no_exp", type=bool, default=False)
    parser.add_argument("--generated_rules", type=bool, default=False)

    args = parser.parse_args()

    for seed in range(42, 47):
        set_random_seed(seed)
        trainer = Trainer(args)
        trainer.train()
