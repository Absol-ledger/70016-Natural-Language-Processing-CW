#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
from urllib import request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DebertaTokenizer, DebertaForSequenceClassification
import os
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup

user="/vol/bitbucket/cx720/cw/nlp/70016-Natural-Language-Processing/"
# user="/vol/bitbucket/xz223/dlenv/NLP/"


# In[2]:


class DontPatronizeMe:

	def __init__(self, _train_path, _test_path):

		self.train_path = _train_path
		self.test_path = _test_path
		self.train_task1_df = None
		self.test_set_df = None

	def load_task1(self):
		"""
		Load task 1 training set and convert the tags into binary labels. 
		Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
		Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
		It returns a pandas dataframe with paragraphs and labels.
		"""
		rows=[]
		with open(self.train_path) as f:
			for line in f.readlines()[4:]:
				par_id=line.strip().split('\t')[0]
				art_id = line.strip().split('\t')[1]
				keyword=line.strip().split('\t')[2]
				country=line.strip().split('\t')[3]
				t=line.strip().split('\t')[4]#.lower()
				l=line.strip().split('\t')[-1]
				if l=='0' or l=='1':
					lbin=0
				else:
					lbin=1
				rows.append(
					{'par_id':par_id,
					'art_id':art_id,
					'keyword':keyword,
					'country':country,
					'text':t, 
					'label':lbin, 
					'orig_label':l
					}
					)
		df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
		self.train_task1_df = df


# In[3]:


def get_test(user):
    _train_path = f'{user}/cw/dontpatronizeme_pcl.tsv'
    _test_path = f'{user}/cw/task4_test.tsv'
    
    dpm = DontPatronizeMe(_train_path, _test_path)
    dpm.load_task1()
    
    train_data = dpm.train_task1_df
    train_data["par_id"] = train_data["par_id"].astype(str)
    
    dev_parids = pd.read_csv("dev_semeval_parids-labels.csv")
    dev_parids["par_id"] = dev_parids["par_id"].astype(str)
    dev_parid_list = dev_parids["par_id"].unique()
    dev_data = train_data[train_data["par_id"].isin(dev_parid_list)]
    return dev_data

def get_train(user):
    _train_path = f'{user}/cw/dontpatronizeme_pcl.tsv'
    _test_path = f'{user}/cw/task4_test.tsv'
    
    dpm = DontPatronizeMe(_train_path, _test_path)
    dpm.load_task1()
    
    train_data = dpm.train_task1_df
    train_data["par_id"] = train_data["par_id"].astype(str)
    
    train_parids = pd.read_csv("train_semeval_parids-labels.csv")
    train_parids["par_id"] = train_parids["par_id"].astype(str)
    train_parid_list = train_parids["par_id"].unique()
    train_filtered_data = train_data[train_data["par_id"].isin(train_parid_list)]
    return train_filtered_data


# In[4]:


train_data = get_train(user)
test_data = get_test(user)


# In[5]:


# split
train_train_data, train_val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['label'])


# In[ ]:


augmenter = naw.ContextualWordEmbsAug(
    model_path = 'distilbert-base-uncased', 
    device = 'cuda',
    action = "substitute",
    top_k = 10
)
train_data_positive = train_train_data.loc[train_train_data["label"]]
augmented_data_all = []

for i in range(len(train_data_positive)):
    original_sentence = train_data_positive.iloc[i]["text"]
    for j in range(5):
        augmented_sentence = augmenter.augment(original_sentence)
        augmented_data = {"keyword": train_data_positive.iloc[i]["keyword"], "text": augmented_sentence[0], "label": True}
        augmented_data_all.append(augmented_data)
augmented_data_df = pd.DataFrame.from_records(augmented_data_all)

train_train_data = pd.concat([train_train_data, augmented_data_df])
train_train_data = train_train_data.sample(frac=1, random_state=1)


# ## Create PyTorch Dataset

# In[6]:


tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base", do_lower_case=False)

def encode_data(data, tokenizer, max_length=512):
    return tokenizer(data['text'].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=max_length)

train_train_encodings = encode_data(train_train_data, tokenizer)
train_val_encodings = encode_data(train_val_data, tokenizer)


# In[7]:


class PCLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[8]:


train_train_dataset = PCLDataset(train_train_encodings, list(train_train_data["label"]))
train_val_dataset = PCLDataset(train_val_encodings, list(train_val_data["label"]))


# ## Compute_metrics

# In[9]:


def compute_metrics(input):
    y_pred = np.argmax(input.predictions, axis=1)
    y_true = input.label_ids
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    return {'accuracy': accuracy, 'f1 score': f1score}


# ## Training

# In[ ]:


from transformers import get_scheduler

learning_rates = [1e-5]

for lr in learning_rates:
    model_name = f"deberta_f1"
    print("Start "+model_name)
    
    num_labels = 2  
    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=num_labels)
    
    device = torch.device('cuda:0')
    
    model.to(device)

    # setting training parameters
    training_args = TrainingArguments(
        output_dir='./Deberta/pcl_deberta_model_base',
        learning_rate=1e-5,  
        weight_decay=0.05,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        do_eval=True,
        evaluation_strategy="epoch",
    )
    
    total_steps = len(train_train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs

    warmup_steps = int(0.1 * total_steps)
    warmup_steps = 0

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_train_dataset,
        eval_dataset=train_val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )
    
    trainer.train()
    with open(f'./outputs/{model_name}.txt', "w") as f:
        for log in trainer.state.log_history:
            if not any(key.startswith("loss") for key in log):
                f.write(str(log) + "\n")
    
    model.save_pretrained(f'./models/{model_name}')
    tokenizer.save_pretrained(f'./tokenizers/{model_name}')

