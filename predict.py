import torch

from model import TextBackbone
from data import ReviewsPredictDataset, load_predict_data

import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Simple Sentiment Predict with PyTorch and Transformers'
    )
    
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')

    parser.add_argument('--data_path', type=str, default='test_data.txt', help='the path of dataset')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    reviews = load_predict_data(args.data_path)
    dataset = ReviewsPredictDataset(reviews)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = TextBackbone(num_classes=args.n_classes).to(args.device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    result = []
    pbar = tqdm(data_loader)
    pbar.set_description("Item")
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(batch)
            output = torch.argmax(output, dim=1).cpu().numpy()
            result.extend(output)

    with open('result.txt', 'w') as f:
        for item in zip(reviews, result):
            f.write('{},{}\n'.format(item[0], item[1]))
            

if __name__ == '__main__':
    main()