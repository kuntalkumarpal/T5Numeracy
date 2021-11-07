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

# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import T5ForConditionalGeneration,T5Tokenizer

import torch.distributed as dist




def compute_exact_match(predicted_answer, correct_answer) -> bool:
    predicted_answer = predicted_answer.strip().lower().replace(" ","")
    correct_answer = correct_answer.strip().lower().replace(" ","")
    return predicted_answer == correct_answer


def value_to_label(predicted_answer):
    
    if predicted_answer < 0 :
        return 8
    if predicted_answer >=0 and predicted_answer < 1:
        return 0
    elif predicted_answer >=1 and predicted_answer < 10:
        return 1
    elif predicted_answer >=10 and predicted_answer < 100:
        return 2
    elif predicted_answer >=100 and predicted_answer < 1000:
        return 3
    elif predicted_answer >=1000 and predicted_answer < 10000:
        return 4
    elif predicted_answer >=10000 and predicted_answer < 100000:
        return 5
    elif predicted_answer >=100000 and predicted_answer < 1000000:
        return 6
    else:
        return 7
    
class T5Finetuner(pl.LightningModule):

    def __init__(self, hparams):
        super(T5Finetuner, self).__init__()
        self.save_hyperparameters()
        
        self.hparams = hparams

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_type)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        
        self.predictions = []
        
        
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def prepare_batch(self, questions, answers):

        input_dict = self.tokenizer.batch_encode_plus(
            list(questions), padding=True, truncation=True, max_length=self.hparams.input_max_len, return_tensors='pt')

        labels = self.tokenizer.batch_encode_plus(
            list(answers), padding=True, truncation=False, max_length=6, return_tensors='pt')['input_ids']

        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)        
        labels = labels.to(self.model.device)

        return input_ids, attention_mask, labels
    
    
    def training_step(self, batch, batch_idx):
        questions = batch['texts']
        answers = batch['mask_answers']
        
        
        input_ids, attention_mask, labels = self.prepare_batch(questions=questions, answers=answers)
        loss = self.model(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          labels=labels, 
#                           decoder_attention_mask=target_mask
                         )
        # loss[0] comes from multiple gpus
        loss = loss[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    
    def inference_step(self, batch, batch_idx):

        texts = batch['texts']
        gold_answers = batch['labels']
        orders = batch['orders']

        input_ids, attention_mask, _ = self.prepare_batch(questions=texts, answers=gold_answers)
        
        batch_outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length
        )

        raw_predicted_answers = [ self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        exact_matches = []
        if self.hparams.em_criteria == "em_raw":
            exact_matches = [ int(compute_exact_match(predicted_answer=e1, correct_answer=e2)) 
                             for e1, e2 in zip(raw_predicted_answers, gold_answers)]
        elif self.hparams.em_criteria == "em_label":
            valid_predicted_answers = [float(fpa) if fpa.replace('.','',1).isdigit() else -99999 for fpa in raw_predicted_answers]
            predicted_labels = [str(value_to_label(each)) for each in valid_predicted_answers ]
            exact_matches = [ int(compute_exact_match(predicted_answer=e1, correct_answer=e2)) 
                                    for e1, e2 in zip(predicted_labels, orders)]
            
        else:
            print("ERROR IN INPUT")
            exit(0)
            
        
        # Log every power of two.
        if batch_idx%500 == 0:
            print('\nQuestion:', texts[0])
            print('Correct:  ', gold_answers[0])
            print('Predictions:', raw_predicted_answers[0])
            print('Exact matches?', exact_matches[0])
            

        metrics = {
                  'exact_matches':exact_matches,
                  'em_sum':sum(exact_matches),
                    }
        return metrics
    
    
    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)
        
    
    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)
    
    
    
    def validation_epoch_end(self, outputs):
        exact_matches = []
        counts = []
        exact_matches_sum = []
            
        for each_batch in outputs:
            
            exact_match_tensor_per_process = torch.tensor(each_batch['em_sum'], dtype=torch.int64)
            all_em_list = [torch.zeros(exact_match_tensor_per_process.size(), dtype=torch.int64).to(self.device) for _ in range(dist.get_world_size())]
            
            dist.all_gather(all_em_list, exact_match_tensor_per_process.to(self.device))
            
            exact_matches_aggr = torch.stack(all_em_list).tolist()
            exact_matches_sum += exact_matches_aggr
        metrics = {'v_em': round(sum(exact_matches_sum)),
                  }
        output = metrics.copy()
        output['progress_bar'] = metrics
        return output
    

    def test_epoch_end(self, outputs):
        exact_matches = []
        counts = []
        exact_matches_sum = []
            
        for each_batch in outputs:
            exact_match_tensor_per_process = torch.tensor(each_batch['em_sum'], dtype=torch.int64)
            
            all_em_list = [torch.zeros(exact_match_tensor_per_process.size(), dtype=torch.int64).to(self.device) for _ in range(dist.get_world_size())]
            
            dist.all_gather(all_em_list, exact_match_tensor_per_process.to(self.device))
            
            exact_matches_aggr = torch.stack(all_em_list).tolist()
            
            exact_matches_sum += exact_matches_aggr
            
        metrics = {'t_em': round(sum(exact_matches_sum))}
        output = metrics.copy()
        output['progress_bar'] = metrics
        return output

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        
        
        texts = batch['texts']
        mask_answers = batch['mask_answers']
        
        
        input_ids, attention_mask, _ = self.prepare_batch(questions=texts, answers=mask_answers)
        
        batch_outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length,
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
        

    

        
        
class Reuters(Dataset):
    def __init__(self, data_dir, type_path, max_len, random_seed, data_size):
        
        self.path = os.path.join(data_dir, type_path + '.jsonl')
        
        
        self.source_column = "masked_text"
        self.target_column = "ans"  # Evaluate - calculate F1, ACC
        self.target_ans = "masked_ans"       # For Loss Calculation
        self.target_order = "label"
            
    

# NUMERACY 600K
# {"id": "20150122163223nI6N0TW027-9", 
# "text": "TOD'S SAYS 2014 PRELIMINARY SALES  AT 965.6 MLN EUROS ", 
# "masked_text": "TOD'S SAYS 2014 PRELIMINARY SALES  AT  <extra_id_0> MLN EUROS ", 
# "ans": 965.6, 
# "masked_ans": "<extra_id_0> 965.6 </s>", 
# "label": 3
# }
        
        self.data = []
        with jsonlines.open(self.path) as f:
            for each in f:
                self.data.append(each)
        if data_size:
            self.data = self.data[:data_size]

        self.texts = []
        self.labels = []
        self.mask_answers = []
        self.orders = []
        
        self._build()

        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]      #original text to track back
        labels = self.labels[index]    #original output to track back
        mask_answers = self.mask_answers[index]
        orders = self.orders[index]
        
        return {"texts": texts, 
                "labels": labels,
                "mask_answers" : mask_answers,
                "orders":orders}

    
    def _build(self):
        ''' Builds the tokenized input and target
        '''
        for idx, each in tqdm(enumerate(self.data)):
            text = each[self.source_column]
            label = str(each[self.target_column])
            mask_ans = each[self.target_ans]
            order = str(each[self.target_order])
            
            
            self.texts.append(text)
            self.labels.append(label)
            self.mask_answers.append(mask_ans)
            self.orders.append(order)
            
            
if __name__ == "__main__":

    stime = time.time()
    
    parser = argparse.ArgumentParser(description='Train and evalute T5 on arithmetic problems.')
    
    parser.add_argument('--data_dir', default="data/.", type=str, required=False, help='Path to take data from.')
    parser.add_argument("--data_size", default=1000, type=int, help="size of data to train with, 0 being Full")
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
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers for loading data.")
    parser.add_argument("--num_beams", default=20, type=int, help="Number of Beams required for decoding.")
    parser.add_argument('--prefix', type=str, default='', help='Prefix of Output')
#     parser.add_argument("--mask_type", default="single_ans_random", type=str, help="Type of Masking")
    parser.add_argument("--input_max_len", default=60, type=int, help="Input length max")
    parser.add_argument('--em_criteria', type=str, default='em_raw', help='em_raw or em_label')

    

    

    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    pl.seed_everything(args.seed)
    
    
    
    
    
    train_data = Reuters( data_dir = args.data_dir, 
                                  type_path="train", 
                                  max_len=args.max_seq_length,
                                  random_seed=args.seed,
                                data_size=args.data_size)
    val_data = Reuters( data_dir = args.data_dir, 
                                  type_path="val", 
                                  max_len=args.max_seq_length,
                                  random_seed=args.seed,
                                data_size=args.data_size)
    
    print("train_data size : ",len(train_data))
    print("val_data size : ",len(val_data))
    
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(train_data[0:5])
    print(val_data[0:5])

    
    
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
    
    