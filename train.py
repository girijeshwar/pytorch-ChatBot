from model import neuralNet
import numpy as np
from bott import tokenize, bag_of_words, stemming
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json',encoding='utf-8') as f:
    intents= json.load(f)

all_words=[]
tags =[]
xy=[]

for intent in intents['intents']:
    tag =intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words=['?','!','.',',']
all_words= [stemming(w) for w in all_words if w not in ignore_words]
tags= sorted(set(tags))


X_train=[]
y_train=[]

for (pat, tag) in xy:
    bag=bag_of_words(pat, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train= np.array(X_train)
y_train= np.array(y_train)


class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data= X_train
        self.y_data = y_train


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples




batch_size=8
hiddensize=8
outputsize= len(tags)
lr = 0.001
num_e= 1300
inputsize= len(X_train[0])

# print(inputsize, len(all_words))
# print(outputsize, tags)
dataset= chatDataset()
train_loader =DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model =neuralNet(inputsize, hiddensize, outputsize).to(device)

criterion= nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(num_e):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_e}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')



data= {
    "model_state":model.state_dict(),
    "input_size": inputsize,
    "output_size": outputsize,
    "hidden_size": hiddensize,
    "all_words": all_words,
    "tags":tags,

}

File ="data.pth"

torch.save(data, File)

print("Training complete. File saved")