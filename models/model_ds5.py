import os
import glob
import spacy
import stanza
import pickle
import pandas as pd
from sklearn.utils import class_weight
from collections import Counter
import torch
torch.cuda.empty_cache()
import numpy as np

#import stanza
#stanza.download('en')
#nlp = stanza.Pipeline(lang='sl', processors='tokenize,pos,lemma')

#import en_core_web_sm
#nlp1 = en_core_web_sm.load()

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
from simpletransformers.ner import NERModel, NERArgs
import pickle as pkl

# P = TP/(TP+FP)
# R =  TP/(TP+FN)
def evaluation_metrics(pred, gt):
  TP = len(set(pred) & set(gt)) 
  FP = len(set(pred)-set(gt))
  FN = len(set(gt)-set(pred))
  print(TP,FP, FN)
  precision = round((TP/(TP+FP))*100, 2)
  recall = round((TP/(TP+FN))*100,2)
  f1_score = round((2 * precision * recall) / (precision + recall),2)
  return precision, recall, f1_score 

groundtruth = pd.read_csv('/Users/andrazrepar/Koda/rsdo/DS5/terminology-extraction/DS5/sl/biomechanics/annotations/biomechanics_sl_terms.ann', sep='	', engine='python',header=None)
gt = list(groundtruth[0]) 

path = "/Users/andrazrepar/Koda/rsdo/DS5/terminology-extraction/DS5/sl/biomechanics/texts/annotated/"
list_of_files = os.listdir("/Users/andrazrepar/Koda/rsdo/DS5/terminology-extraction/DS5/sl/biomechanics/texts/annotated/")
lines=[]
for file in list_of_files:
    if not file.startswith('.'):
        f = open(path+file, "r")
        #append each line in the file to a list
        lines.append(f.readlines())
        f.close()

    


train = pd.read_csv('/Users/andrazrepar/Koda/rsdo/DS5/terminology-extraction/processed_data/slann_train_inl.csv')
train['words'] = [str(x) for x in train['words']]
train_df, eval_df = train_test_split(train, test_size=0.15, random_state=2021)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_df.labels),
                                                 train_df.labels)


model_args = NERArgs(
                    labels_list = ['B','I', 'O'],
                    manual_seed = 2021,
                    num_train_epochs = 4,
                    max_seq_length = 512,
                    use_early_stopping = True,
                    overwrite_output_dir = True,
                    train_batch_size = 8
                    )

model  = NERModel(
    #"bert", "EMBEDDIA/crosloengual-bert", args=model_args, weight = [2.52853895, 12.18978944,  0.39643544], use_cuda=False, cuda_device=-1
    "bert", "EMBEDDIA/crosloengual-bert", args=model_args, use_cuda=False, cuda_device=-1
)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(
    eval_df
)

terms = []
preds = []
##for lines_ in lines:
#  sentences = [[token.text for token in nlp1(line.strip())] for line in lines_]
#  predictions, raw_outputs = model.predict(sentences, split_on_space=False)
#  preds.extend(predictions)

with open('1ann_weighted_roberta.pkl', 'wb') as f:
    pkl.dump(preds, f)