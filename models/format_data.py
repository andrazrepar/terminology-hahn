import pandas as pd
import pickle
from tqdm import tqdm

train = ['/Users/andrazrepar/Koda/rsdo/DS5/terminology-extraction/processed_data/slann_train_inl.pkl']
train_df = pd.DataFrame(columns=["sentence_id", "words", "labels"])
final_train_df = pd.DataFrame()

for i in train:
    with open(i, "rb") as input_file:
        sentences, labels, tokens, terms = pickle.load(input_file)
    sentence_id = []
    words = []
    targets = []

    for index, (token, label) in tqdm(enumerate(zip(tokens, labels))):
        for t, l in zip(token, label):
            sentence_id.append(index)
            words.append(t)
            targets.append(l)
    train_df['sentence_id'] = sentence_id
    train_df['words'] = words
    train_df['labels'] = targets
    final_train_df = final_train_df.append(train_df, ignore_index=True)

print(final_train_df.labels.value_counts())
final_train_df.to_csv('/Users/andrazrepar/Koda/rsdo/DS5/terminology-extraction/processed_data/slann_train_inl.csv', index=False)