import torch
from torch.nn.functional import cross_entropy

import argparse
from tqdm import tqdm

from model import TextBackbone
from data import ReviewsDataset, load_data

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Simple Sentiment Analysis with PyTorch and Transformers'
    )
    
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')

    parser.add_argument('--data_path', type=str, default='data/data.txt', help='the path of dataset')
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')

    parser.add_argument('--epochs', default=50, type=int, help='number of epochs tp train for')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def train(model, dataset, optimizer, device, batch_size, epochs):
    model.train()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    pbar = tqdm(train_loader)
    for epoch in range(epochs):
        pbar.set_description("Epoch {}:".format(epoch))
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(batch)
            loss = cross_entropy(output, batch['targets'])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        
        if epoch % 10 == 0:
            model.save('model_{}.pth'.format(epoch))

    return model
    

def main():
    args = parse_arguments()
    reviews, targets = load_data(args.data_path)
    dataset = ReviewsDataset(reviews, targets)
    model = TextBackbone(num_classes=args.n_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = train(model, dataset, optimizer, args.device, args.batch_size, args.epochs)
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()