import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from sub_model import TimeDistributed
from progressbar import *

class PULSTMCNN(nn.Module):
    def __init__(self, dp, charModel, wordModel, caseModel, featureModel, inputSize, hiddenSize, layerNum, dropout):
        super(PULSTMCNN, self).__init__()
        self.dp = dp
        self.charModel = TimeDistributed(charModel, self.dp.char2Idx)
        self.wordModel = wordModel
        self.caseModel = caseModel
        self.featureModel = featureModel
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=layerNum,batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hiddenSize, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Softmax(dim=2)
            # nn.Linear(200, 1)
        )

    def forward(self, token, case, char, feature):
        charOut, sortedLen1, reversedIndices1 = self.charModel(char)
        wordOut, sortedLen2, reversedIndices2 = self.wordModel(token)
        caseOut, sortedLen3, reversedIndices3 = self.caseModel(case)
        featureOut, sortedLen4, reversedIndices4 = self.featureModel(feature)

        encoding = torch.cat([wordOut.float(), caseOut.float(), charOut.float(), featureOut.float()], dim=2)

        sortedLen = sortedLen1
        reverseIndices = reversedIndices1

        packed_embeds = pack_padded_sequence(encoding, sortedLen, batch_first=True)

        maxLen = sortedLen[0]
        mask = torch.zeros([len(sortedLen), maxLen, 2])
        for i, l in enumerate(sortedLen):
            mask[i][:l][:] = 1

        lstmOut, (h, _) = self.lstm(packed_embeds)

        paddedOut = pad_packed_sequence(lstmOut, batch_first=True)

        # print(paddedOut)

        fcOut = self.fc(paddedOut[0])

        fcOut = fcOut * mask.cuda()
        fcOut = fcOut[reverseIndices]

        return fcOut

    def loss_func(self, yTrue, yPred, type):
        loss = 0
        y = torch.eye(2)[yTrue].float().cuda()
        if len(y.shape) == 1:
            y = y[None, :]
        # y = torch.from_numpy(yTrue).float().cuda()
        if type == 'bnpu' or type == 'bpu':
            loss = torch.mean((y * (1 - yPred)).sum(dim=1))
        elif type == 'upu':
            loss = torch.mean((-y * torch.log(yPred)).sum(dim=1))
        # loss = 0.5 * torch.max(1-yPred*(2.0*yTrue-1),0)
        return loss


class Trainer(object):
    def __init__(self, model, prior, beta, gamma, learningRate, m):
        self.model = model
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.learningRate,
                                            weight_decay=1e-8)
        self.m = m
        self.prior = prior
        self.bestResult = 0
        self.beta = beta
        self.gamma = gamma
        self.positive = np.eye(2)[1]
        self.negative = np.eye(2)[0]

    def train_mini_batch(self, batch, args):
        token, case, char, feature, label, flag = batch
        length = [len(i) for i in flag]
        maxLen = max(length)
        fids = []
        lids = []
        for s in flag:
            f = list(s)
            f += [np.array([-1, -1]) for _ in range(maxLen - len(f))]
            fids.append(f)
        for s in label:
            l = list(s)
            l += [np.array([-1, -1]) for _ in range(maxLen - len(l))]
            lids.append(l)
        fids = np.array(fids)
        lids = np.array(lids)

        postive = (fids == self.positive) * 1
        unlabeled = (fids == self.negative) * 1

        self.optimizer.zero_grad()
        result = self.model(token, case, char, feature)

        hP = result.masked_select(torch.from_numpy(postive).bool().cuda()).contiguous().view(-1, 2)
        hU = result.masked_select(torch.from_numpy(unlabeled).bool().cuda()).contiguous().view(-1, 2)
        if len(hP) > 0:
            pRisk = self.model.loss_func(1, hP, args.type)
        else:
            pRisk = torch.FloatTensor([0]).cuda()
        uRisk = self.model.loss_func(0, hU, args.type)
        nRisk = uRisk - self.prior * (1 - pRisk)
        risk = self.m * pRisk + nRisk

        if args.type == 'bnpu':
            if nRisk < self.beta:
                risk = -self.gamma * nRisk
        # risk = self.model.loss_func(label, result)
        new_risk = risk.clone().detach().requires_grad_(True)
        new_risk.backward()
        # torch.tensor((risk), requires_grad=True).backward()
        self.optimizer.step()
        pred = torch.argmax(hU, dim=1)
        label = torch.tensor(lids, dtype=torch.long, device='cuda')

        unlabeledY = label.masked_select(torch.from_numpy(unlabeled).bool().cuda()).contiguous().view(-1, 2)

        acc = torch.mean((torch.argmax(unlabeledY, dim=1) == pred).float())
        return acc, new_risk, pRisk, nRisk

    def test(self, batch, length):
        token, case, char, feature = batch
        maxLen = max([x for x in length])
        mask = np.zeros([len(token), maxLen, 2])
        for i, x in enumerate(length):
            mask[i][:x][:] = 1
        result = self.model(token, case, char, feature)
        # print(result)
        result = result.masked_select(torch.from_numpy(mask).bool().cuda()).contiguous().view(-1, 2)
        pred = torch.argmax(result, dim=1)

        temp = result[:, 1]
        return pred.cpu().numpy(), temp.detach().cpu().numpy()

    def save(self, dir):
        if dir is not None:
            torch.save(self.model.state_dict(), dir)

    def decay_learning_rate(self, epoch, init_lr):
        """衰减学习率

        Args:
            epoch: int, 迭代次数
            init_lr: 初始学习率
        """
        lr = init_lr / (1 + 0.05 * epoch)
        print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer