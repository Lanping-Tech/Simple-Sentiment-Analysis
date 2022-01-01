import torch
from torch.nn.functional import cross_entropy

import argparse
from tqdm import tqdm

from model import TextBackbone
from data import ReviewsDataset, load_data

from transformers import AdamW

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Simple Sentiment Analysis with PyTorch and Transformers'
    )
    
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')

    parser.add_argument('--data_path', type=str, default='data/data.txt', help='the path of dataset')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')

    parser.add_argument('--epochs', default=50, type=int, help='number of epochs tp train for')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def train(model, dataset, optimizer, device, batch_size, epochs):
    model.train()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    min_loss = float('inf')
    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        pbar.set_description("Epoch {}:".format(epoch))
        total_loss = 0
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            loss = cross_entropy(output, batch['targets'])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            total_loss += loss.item()
        
        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(model.state_dict(), 'output/model_best.pth')

        print("Epoch {}: Average loss: {}".format(epoch, total_loss / len(train_loader)))

    return model
    

def main():
    args = parse_arguments()
    reviews, targets = load_data(args.data_path)
    dataset = ReviewsDataset(reviews, targets)
    model = TextBackbone(num_classes=args.n_classes).to(args.device)
    optimizer = AdamW(model.parameters(),lr=2e-5, eps=1e-8)

    model = train(model, dataset, optimizer, args.device, args.batch_size, args.epochs)
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()