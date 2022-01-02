import torch
import torch.nn.functional as F

from model import TextBackbone
from data import ReviewsPredictDataset, load_predict_data

import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Simple Sentiment Predict with PyTorch and Transformers'
    )
    
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')

    parser.add_argument('--data_path', type=str, default='output/comments.csv', help='the path of dataset')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def load_specific_data(data_path):
    data = pd.read_csv(data_path)
    comments = data[['space_comment','power_comment','control_comment','fuel_comment','comfortable_comment','outer_comment','inner_comment','quality_comment']]
    comments = [item for sublist in comments.values.tolist() for item in sublist]
    result = []
    for comment in comments:
        comment = str(comment)
        if comment == 'nan':
            comment = '一般'
        comment = comment.replace('\n', '').replace('\r', '').replace('\t', '')
        if '：' in comment:
            comment = comment.split('：')[1]
        result.append(comment)
    return result


def main():
    args = parse_arguments()
    reviews = load_specific_data(args.data_path)
    dataset = ReviewsPredictDataset(reviews)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = TextBackbone(num_classes=args.n_classes).to(args.device)
    model.load_state_dict(torch.load('output/model_best.pth', map_location=args.device))
    model.eval()
    result = []
    pbar = tqdm(data_loader)
    pbar.set_description("Item")
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(batch)
            output = F.softmax(output)
            value = torch.max(output, dim=1)[0].cpu().numpy()
            result.extend(value)

    result = np.array(result).reshape(-1,8)

    result.tofile('comment_vector.csv',sep=',',format='%10.5f')

    # with open('comment_vector.csv', 'w') as f:
    #     for item in zip(reviews, result):
    #         f.write('{},{}\n'.format(item[0], item[1]))
            

if __name__ == '__main__':
    main()