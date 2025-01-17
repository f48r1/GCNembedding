import numpy as np
from tqdm import tqdm

import torch
from torch.nn import Linear, Sigmoid
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, h1=10, h2=5, output=2, lr=0.001, inputDim=None):
        super(GCN,self).__init__()
        self.lr = lr
        self.output = output
        torch.manual_seed(42)
        self.conv1 = GCNConv(inputDim,h1)
        self.conv2 = GCNConv(h1,h2)
        self.conv3 = GCNConv(h2,output)
        self.classfier = Linear(output,output)
        self.sigm = Sigmoid()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.classfier(h)
        if self.output==1:
            out = self.sigm(out)
        return out, h
    
    def trainModel(self, data, epochs, mask=None, bar = None):

        if type(mask) == np.ndarray:
            self._maskNodes = mask
            self._maskEdges = np.isin(data.edge_index, mask).all(axis=0)
        else:
            self._maskEdges = slice(None)
            self._maskNodes = slice(None)
        
        self.opt = torch.optim.Adam(self.parameters(),lr=self.lr)
        self.crit = torch.nn.CrossEntropyLoss() if self.output>1 else torch.nn.BCELoss()

        embeddings = []
        losses = []

        if not bar:
            bar = tqdm(range(1,epochs+1), desc="Epochs")

        for epoch in bar:
            loss, h = self._train(data)
            losses.append(float(loss))
            embeddings.append(h)
        return embeddings, losses

    def _train(self, data):
        self.opt.zero_grad()
        # out, h = self(data.x,data.edge_index[:,self._maskEdges])
        out, h = self(data.x, data.edge_index)

        if self.output == 1:
            loss = self.crit(out[self._maskNodes], data.y[self._maskNodes].view(-1,1).to(dtype=torch.float32))
        else:
            loss = self.crit(out[self._maskNodes], data.y[self._maskNodes])
        
        loss.backward()
        self.opt.step()
        
        return loss, h
