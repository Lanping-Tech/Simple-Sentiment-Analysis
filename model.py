from re import X
import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import pipeline

class TextBackbone(torch.nn.Module):

    def __init__(self, pretrained_model_name='chinese-roberta-wwm-ext', num_classes=2):
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained_model_name)
        self.fc = torch.nn.Linear(768, num_classes)


    def forward(self, x):
        inputs = {'input_ids':x['input_ids'],'attention_mask':x['attention_mask'],'token_type_ids':x['token_type_ids']}
        outputs = self.extractor(**inputs)
        out = self.fc(outputs.pooler_output)
        return out


if __name__ == '__main__':
    model = TextBackbone()
    x = ['我爱学习', '我不爱学习']
    tokenizer = AutoTokenizer.from_pretrained('chinese-roberta-wwm-ext')
    x = tokenizer.encode_plus(x, max_length=512, padding='max_length', return_tensors='pt')
    y = model(x)
    print(y.shape)