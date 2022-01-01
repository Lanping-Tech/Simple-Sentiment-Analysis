import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class ReviewsDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer_name='chinese-roberta-wwm-ext', max_len=512):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.len = len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt')

        return {
            # 'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids' : encoding['token_type_ids'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

    def __len__(self):
        return self.len

class ReviewsPredictDataset(Dataset):
    def __init__(self, reviews, tokenizer_name='chinese-roberta-wwm-ext', max_len=512):
        self.reviews = reviews
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.len = len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt')

        return {
            # 'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids' : encoding['token_type_ids'].flatten(),
        }

    def __len__(self):
        return self.len

def load_data(file_path):
    reviews = []
    targets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            review, target = line.strip().split('\t')
            reviews.append(review)
            targets.append(int(target))
    return reviews, targets

def load_predict_data(file_path):
    reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            review = line.strip()
            reviews.append(review)
    return reviews

if __name__ == '__main__':
    reviews, targets = load_data('test_data.txt')
    dataset = ReviewsDataset(reviews, targets)
    print(dataset[0])