class CustomDataset(Dataset):
    def init(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len 
    def getitem(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
        sentence,
        None,
        add_special_tokens=True,
        max_length=self.max_len,
        pad_to_max_length=True,
        return_token_type_ids=True
        )
        ids = inputs['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([4]*200)
        label=label[:200]
        #print(sentence)
        #print(self.tokenizer.convert_ids_to_tokens(ids))
        #print(ids)
        #print(label) map_idx = 0
        new_labels = [] 
        for i in range(len(tokens)):
            if ids[i] in [101, 102]:
                map_idx += 1
                new_labels.append(0.0)
            elif tokens[i].startswith('##'):
                map_idx += 1
                new_labels.append(label[i - map_idx])
            else:
                new_labels.append(label[i - map_idx])
                label = new_labels
        return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'tags': torch.tensor(label, dtype=torch.float)
    } 
    def len(self):
        return self.len