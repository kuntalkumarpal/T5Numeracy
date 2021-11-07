import argparse
import glob
import json
import os
import pytorch_lightning as pl
import random
import torch
import time
import jsonlines
from tqdm import tqdm
import shutil
import itertools
import re

# from num2words import num2words
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import TFT5ForConditionalGeneration

# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import T5ForConditionalGeneration,T5Tokenizer

import torch.distributed as dist




def compute_exact_match(predicted_answer, correct_answer) -> bool:
    predicted_answer = predicted_answer.strip().lower().replace(" ","")
    correct_answer = correct_answer.strip().lower().replace(" ","")
    return predicted_answer == correct_answer


class T5Finetuner(pl.LightningModule):

    def __init__(self, hparams):
        super(T5Finetuner, self).__init__()
        self.save_hyperparameters()
        
        self.hparams = hparams
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_type)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name_or_path)

        
        self.predictions = []
        
        
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def prepare_batch(self, questions, answers):

        input_dict = self.tokenizer.batch_encode_plus(
            list(questions), padding=True, truncation=True, return_tensors='pt')

        labels = self.tokenizer.batch_encode_plus(
            list(answers), padding=True, truncation=False, return_tensors='pt')['input_ids']

        
        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)
        labels = labels.to(self.model.device)

        return input_ids, attention_mask, labels
    
    
    def training_step(self, batch, batch_idx):
        
        questions = batch['texts']
        answers = batch['labels']
        
        input_ids, attention_mask, labels = self.prepare_batch(questions=questions, answers=answers)
        loss = self.model(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          labels=labels, 
#                           decoder_attention_mask=target_mask
                         )
        # loss[0] comes from multiple gpus
        loss = loss[0]
#         print("loss:training_step:\n",loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def inference_step(self, batch, batch_idx):

        texts = batch['texts']
        gold_answers = batch['labels']

        input_ids, attention_mask, _ = self.prepare_batch(questions=texts, answers=gold_answers)
        
        batch_outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length,
#             num_beams=self.hparams.num_beams
        )

        preds = [ self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]
        
        exact_matches = [ int(compute_exact_match(predicted_answer=e1, correct_answer=e2)) 
                         for e1, e2 in zip(preds, gold_answers)]
        
        # Log every power of two.
        if batch_idx%100 == 0:
            print('\nQuestion:', texts[0])
            print('Correct:  ', gold_answers[0])
            print('preds:', preds[0])
            print('Exact-Filter?', exact_matches[0])
            

        metrics = {
                  'eml_len':len(exact_matches),
                  'exact_matches':exact_matches,
                  'em_sum':sum(exact_matches)}
        return metrics
    
    
    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)
        
    
    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)
    
    
    
    def validation_epoch_end(self, outputs):
        exact_matches = []
        counts = []
        exact_matches_sum = []
        exact_matches_len = []
        exact_matches_label_sum = []
        exact_matches_raw_sum = []
            
        for each_batch in outputs:
            exact_match_len_tensor_per_process = torch.tensor(each_batch['eml_len'], dtype=torch.int64)
            exact_match_tensor_per_process = torch.tensor(each_batch['em_sum'], dtype=torch.int64)
            
            all_em_list = [torch.zeros(exact_match_tensor_per_process.size(), dtype=torch.int64).to(self.device) for _ in range(dist.get_world_size())]
            all_em_len_list = [torch.zeros(exact_match_len_tensor_per_process.size(), dtype=torch.int64).to(self.device) for _ in range(dist.get_world_size())]
            
            
            
            dist.all_gather(all_em_list, exact_match_tensor_per_process.to(self.device))
            dist.all_gather(all_em_len_list, exact_match_len_tensor_per_process.to(self.device))
            
            exact_matches_aggr = torch.stack(all_em_list).tolist()
            exact_matches_len_aggr = torch.stack(all_em_len_list).tolist()

            exact_matches_sum += exact_matches_aggr
            exact_matches_len += exact_matches_len_aggr

            metrics = {'v_em': round(sum(exact_matches_sum)), 
                   'v_cnt':sum(exact_matches_len),
                  }
        output = metrics.copy()
        output['progress_bar'] = metrics
        return output
    

    def test_epoch_end(self, outputs):
        exact_matches = []
        counts = []
        exact_matches_sum = []
        exact_matches_len = []
        exact_matches_label_sum = []
            
        for each_batch in outputs:
            exact_match_len_tensor_per_process = torch.tensor(each_batch['eml_len'], dtype=torch.int64)
            exact_match_tensor_per_process = torch.tensor(each_batch['em_sum'], dtype=torch.int64)
            
            all_em_list = [torch.zeros(exact_match_tensor_per_process.size(), dtype=torch.int64).to(self.device) for _ in range(dist.get_world_size())]
            all_em_len_list = [torch.zeros(exact_match_len_tensor_per_process.size(), dtype=torch.int64).to(self.device) for _ in range(dist.get_world_size())]
            
            
            dist.all_gather(all_em_list, exact_match_tensor_per_process.to(self.device))
            dist.all_gather(all_em_len_list, exact_match_len_tensor_per_process.to(self.device))
            
            exact_matches_aggr = torch.stack(all_em_list).tolist()
            exact_matches_len_aggr = torch.stack(all_em_len_list).tolist()
            
            exact_matches_sum += exact_matches_aggr
            exact_matches_len += exact_matches_len_aggr
            
        metrics = {'t_em': round(sum(exact_matches_sum)), 't_cnt':sum(exact_matches_len)}
        output = metrics.copy()
        output['progress_bar'] = metrics
        return output

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        
        
        texts = batch['texts']
        gold_answers = batch['labels']
        
#         print(texts)
#         print(gold_answers)
#         print(em_answers)
#         print(mask_answers)
        
        input_ids, attention_mask, _ = self.prepare_batch(questions=texts, answers=gold_answers)
        
        batch_outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length,
#             num_beams=self.hparams.num_beams
            )


        raw_predicted_answers = [ self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]
        
            
            
        return raw_predicted_answers

    
    def predict(self, *args, **kwargs):  # for version 1.2.6
        return self.predict_step(*args, **kwargs)
    
    
    
    def get_optimizer(self):
        optimizer_name = self.hparams.optimizer
        scheduler_name = self.hparams.scheduler
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        optimizer = getattr(torch.optim, optimizer_name)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = optimizer(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        print(f'=> Using {optimizer_name} optimizer')

        if scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
            print(f'=> Using StepLR (step_size = {self.hparams.step_size}, gamma = {self.hparams.gamma})')
        else:
            raise Exception(f'Scheduler not implemented: {scheduler_name}')

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer
        

    

        
        
class MinMax(Dataset):
    def __init__(self, data_dir, type_path, max_len, random_seed, data_size, representation):
        
        self.path = os.path.join(data_dir, type_path + '.jsonl')
        self.source_column = ""
        self.target_column = ""
        if representation == "min":
            self.target_column = "min_ans"  # Evaluate - calculate F1, ACC
            self.source_column = "min_text"
        elif representation == "max":
            self.target_column = 'max_ans'
            self.source_column = "max_text"
        elif representation == "asc":
            self.target_column = 'asc_ans'
            self.source_column = "asc_text"
        elif representation == "desc":
            self.target_column = 'desc_ans'
            self.source_column = "desc_text"
        else:
            print("INVALID representation. EXIT")
            exit(0)
            
#             {"numbers": [22, 87, 64], "max_text": "Which is maximum in value among 22 87 64 ?", "max_ans": "87", "min_text": "Which is minimum in value among 22 87 64 ?", "min_ans": "22", "asc_ans": "22 64 87", "desc_ans": "87 64 22", "asc_text": "Sort in ascending order : 22 87 64", "desc_text": "Sort in descending order : 22 87 64"}
        
        self.data = []
        with jsonlines.open(self.path) as f:
            for each in f:
                self.data.append(each)
        if data_size:
            self.data = self.data[:data_size]

        self.texts = []
        self.labels = []
        
        self._build()

        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]      #original text to track back
        labels = self.labels[index]    #original output to track back
        
        return {"texts": texts, 
                "labels": labels}

    
    def _build(self):
        ''' Builds the tokenized input and target
        '''
        for idx, each in tqdm(enumerate(self.data)):
            text = each[self.source_column]
            label = str(each[self.target_column])
            
            
            self.texts.append(text)
            self.labels.append(label)
            
            
#             print(text)
#             print(label)
#             input()
            
if __name__ == "__main__":

    stime = time.time()
    
    parser = argparse.ArgumentParser(description='Train and evalute T5 on arithmetic problems.')
    
    parser.add_argument('--data_dir', default=".", type=str, required=False, help='Path to take data from.')
    parser.add_argument("--data_size", default=0, type=int, help="size of data to train with, 0 being Full")
    parser.add_argument('--output_dir', default="./models/test", type=str, required=False, help='Path to save checkpoint and results.')
    parser.add_argument('--model_name_or_path', default="t5-small", type=str, required=False)
    parser.add_argument('--model_type', default="t5-small", type=str, required=False)
    
    
    parser.add_argument("--seed", default=42, type=int, help="Seed.")
    parser.add_argument("--train_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument("--val_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument("--test_size", default=2000, type=int, help="Number of examples for testing.")
    parser.add_argument('--max_seq_length', type=int, default=20, help='Maximum sequence length (in tokens).')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        help='learning rate scheduler. Currently, only StepLR is supported.)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma factor for ExponentialLR or StepLR')
    parser.add_argument('--step_size', type=int, default=2, help='period of learning rate decay (StepLR)')
    parser.add_argument('--t_0', type=int, default=2,
                        help='number of iterations for the first restart (CosineAnnealingWarmRestarts)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='a factor increases t_i after a restart (CosineAnnealingWarmRestarts)')
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")
    parser.add_argument("--num_beams", default=20, type=int, help="Number of Beams required for decoding.")
    parser.add_argument('--prefix', type=str, default='', help='Prefix of Output')
    parser.add_argument("--representation", default="split_digit", type=str, help="Type of Number Representations : split_digit or full")
    

    

    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    pl.seed_everything(args.seed)
    
    
    
    
    
    train_data = MinMax( data_dir = args.data_dir, 
                                  type_path="train", 
                                  max_len=args.max_seq_length,
                                  random_seed=args.seed,
                                data_size=args.data_size,
                                representation=args.representation)
    val_data = MinMax( data_dir = args.data_dir, 
                                  type_path="val", 
                                  max_len=args.max_seq_length,
                                  random_seed=args.seed,
                                data_size=args.data_size,
                                representation=args.representation)
    test_data = MinMax( data_dir = args.data_dir, 
                                  type_path="test", 
                                  max_len=args.max_seq_length,
                                  random_seed=args.seed,
                                data_size=args.data_size,
                                representation=args.representation)
    expol_test_data = MinMax( data_dir = args.data_dir, 
                                  type_path="expol_test", 
                                  max_len=args.max_seq_length,
                                  random_seed=args.seed,
                                data_size=args.data_size,
                                representation=args.representation)
    
    print("train_data size : ",len(train_data))
    print("val_data size : ",len(val_data))
    print("test_data size : ",len(test_data))
    print("expol_test_data size : ",len(expol_test_data))
    
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    expol_test_dataloader = DataLoader(expol_test_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(train_data[0:5])
    print(val_data[0:5])
    print(test_data[0:5])
    print(expol_test_data[0:5])

    
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir),
        verbose=False, save_last=False, save_top_k=1, mode='max', monitor='v_em',
        save_weights_only=False, period=args.check_val_every_n_epoch,
    )
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, deterministic=True)

    model = T5Finetuner(hparams=args)
    trainer.fit(model, train_dataloader, val_dataloader)
    
    print("Training Time:",round((time.time()-stime)/3600,2))
    print('Best path: ', checkpoint_callback.best_model_path)  # None
    print('Best score: ', checkpoint_callback.best_model_score)  # 0
    
    
    
    results = trainer.test(model, test_dataloader)
    print("Test Results:", results)
    
    results_expol = trainer.test(model, expol_test_dataloader)
    print("Test Results Expol:", results_expol)
    
    
    #### --- PREDICTING INPOL -----
    
    args.gpus = 1
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True)
#     model = T5Finetuner(hparams=args)
    preds = trainer.predict(model, test_dataloader)
    print(len(preds))
    
    
    labels = [each['labels'] for each in test_data]
    texts =  [each['texts'] for each in test_data]
    
    pred = [each_result for each_batch in preds for each_result in each_batch]
    em = [int(compute_exact_match(ep, el)) for ep, el in zip(pred, labels)]
    print("Inpol lengths : ",len(labels),len(texts),len(pred))
    
    

    with jsonlines.open(os.path.join(args.output_dir, args.prefix+'inpol_pred.jsonl'), 'w') as fout:
        for t,l,p,e in zip(texts,labels,pred, em):
            d = {"text":t,
                 "label":l,
                 "prediction":p,
                 "em":e
                }
            fout.write(d)
    
    
    
    #### --- PREDICTING EXPOL -----
    
    args.gpus = 1
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True)
#     model = T5Finetuner(hparams=args)
    preds = trainer.predict(model, expol_test_dataloader)
    print(len(preds))
    
    
    labels = [each['labels'] for each in expol_test_data]
    texts =  [each['texts'] for each in expol_test_data]
    
    pred = [each_result for each_batch in preds for each_result in each_batch]
    em = [int(compute_exact_match(ep, el)) for ep, el in zip(pred, labels)]
    print("Expol lengths : ",len(labels),len(texts),len(pred))
    


    with jsonlines.open(os.path.join(args.output_dir, args.prefix+'expol_pred.jsonl'), 'w') as fout:
        for t,l,p,e in zip(texts,labels,pred, em):
            d = {"text":t,
                 "label":l,
                 "prediction":p,
                 "em":e
                }
            fout.write(d)