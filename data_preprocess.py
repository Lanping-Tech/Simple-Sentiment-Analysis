import os

reviews = []
with open('data/positive.txt', 'r') as f:
    for line in f:
        line = line.strip().replace('\t', '。').replace('\n', '').replace(',', '，')
        reviews.append(line+'\t0')

with open('data/negative.txt', 'r') as f:
    for line in f:
        line = line.strip().replace('\t', '。').replace('\n', '').replace(',', '，')
        reviews.append(line+'\t1')

with open('data/data.txt', 'w') as f:
    for review in reviews:
        f.write(review+'\n')