import pickle as pkl
import pandas as pd


def evaluation_metrics(pred, gt):
    TP = len(set(pred) & set(gt)) 
    FP = len(set(pred)-set(gt))
    FN = len(set(gt)-set(pred))
    precision = round((TP/(TP+FP))*100, 2)
    recall = round((TP/(TP+FN))*100,2)
    f1_score = round((2 * precision * recall) / (precision + recall),2)
    return precision, recall, f1_score 

def get_term_(predictions):
    all_term = []
    for sentence in predictions:
        tokens = []
        labels = []
        for d in sentence:
            tokens.extend(d.keys())
            labels.extend(d.values())

        for i, label in enumerate(labels):
            if labels[i] == 'I' and (i == 0 or labels[i - 1] == 'O'):
                labels[i] = 'O'

        terms = []
        term = []
        for token, label in zip(tokens, labels):
            if label == 'B':
                #Lưu vị trí B
                b_pos = i
                term = [token]
            elif label == 'I':
                term.append(token)
            elif len(term) > 0:
                terms.append(' '.join(term))
                term = []
        if len(term) > 0:
            terms.append(' '.join(term))
            # Check b_pos = 0 không
        all_term.append(terms)
    
    final_terms = []
    for i in all_term:
        final_terms.extend(i)

    final_terms = [x.lower().strip() for x in final_terms]
    return final_terms 

def length_filter(predictions):
    filtered_preds = []
    for pred in predictions:
        if len(pred) < 3: #remove all predicted terms shorter than n characters
            continue
        else:
            filtered_preds.append(pred)
    return filtered_preds

def underscore_filter(predictions):
    filtered_preds = []
    for pred in predictions:
        if '_' in pred: #remove all predicted terms shorter than n characters
            continue
        else:
            filtered_preds.append(pred)
    return filtered_preds

def term_evaluation(domain_path, preds_path, rule=None):
    groundtruth = pd.read_csv(domain_path, sep='	', engine='python',header=None)
    gt = list(groundtruth[0])
    predictions = pkl.load(open(preds_path, 'rb'))
    preds =  get_term_(predictions)
    clean_preds = length_filter(preds)
    clean_preds = underscore_filter(clean_preds)


    print("All predictions:", len(preds))
    print("Cleaned predictions:", len(clean_preds))
    precision, recall, f1 = evaluation_metrics(clean_preds, gt)
    return precision, recall, f1, clean_preds

#predictions = pkl.load( open( "/Users/andrazrepar/Koda/rsdo/DS5/terminology-hahn/processed_data/sl/crosloenbert_ds5_correct_weights.pkl", "rb" ) )

precision, recall, f1, preds = term_evaluation('/Users/andrazrepar/Koda/rsdo/DS5/terminology-hahn/DS5/sl/biomechanics/annotations/biomechanics_sl_terms.ann', '/Users/andrazrepar/Koda/rsdo/DS5/terminology-hahn/processed_data/sl/sloberta_ds5_correct_weights.pkl')
print(precision, recall, f1)
unique_terms = []
for term in preds:
    if term not in unique_terms:
        unique_terms.append(term)

with open('/Users/andrazrepar/Koda/rsdo/DS5/terminology-hahn/processed_data/sl/predictions-sloberta-underscore-length3.txt', 'w') as wf:
    for term in unique_terms:
        wf.write(term + '\n')