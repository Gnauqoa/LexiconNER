import torch
import numpy as np
import argparse
from utils.data_utils import DataPrepare
from utils.feature_pu_model_utils import FeaturedDetectionModelUtils
from sub_model import CharCNN, CaseNet, WordNet, FeatureNet
from progressbar import *
import feature_pu_model as feature
import os
parser = argparse.ArgumentParser(description="PU NER")
# data
parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
parser.add_argument('--beta', type=float, default=0.0,help='beta of pu learning (default 0.0)')
parser.add_argument('--gamma', type=float, default=1.0,help='gamma of pu learning (default 1.0)')
parser.add_argument('--drop_out', type=float, default=0.5, help = 'dropout rate')
parser.add_argument('--m', type=float, default=0.3, help='class balance rate')
parser.add_argument('--flag', default="PER" , help='entity type (PER/LOC/ORG/MISC)')
parser.add_argument('--dataset', default="conll2003",help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=64,help='batch size for training and testing')
parser.add_argument('--epochs', type=int, default=10,help='epochs for printing result')
parser.add_argument('--pert', type=float, default=1.0,help='percentage of data use for training')
parser.add_argument('--type', type=str, default='bnpu',help='pu learning type (bnpu/bpu/upu)')  # bpu upu

args = parser.parse_args()

dp = DataPrepare(args.dataset)
mutils = FeaturedDetectionModelUtils(dp)

trainSet, validSet, testSet, prior = mutils.load_dataset(args.flag, args.dataset, args.pert)

trainSize = len(trainSet)
validSize = len(validSet)
testSize = len(testSet)
print(("train set size: {}, valid set size: {}, test set size: {}").format(trainSize, validSize, testSize))

charcnn = CharCNN(dp.char2Idx)
wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)
featurenet = FeatureNet()
pulstmcnn = feature.PULSTMCNN(dp, charcnn, wordnet, casenet, featurenet, 150, 200, 1, args.drop_out)

if torch.cuda.is_available:
    charcnn.cuda()
    wordnet.cuda()
    casenet.cuda()
    featurenet.cuda()
    pulstmcnn.cuda()
    torch.cuda.manual_seed(1013)

trainer = feature.Trainer(pulstmcnn, prior, args.beta, args.gamma, args.lr, args.m)

time = 0

bar = ProgressBar(maxval=int((len(trainSet) - 1) / args.batch_size))

train_sentences = dp.read_origin_file("data/" + args.dataset + "/train.txt")
trainSize = int(len(train_sentences) * args.pert)
train_sentences = train_sentences[:trainSize]
train_words = []
train_efs = []
for s in train_sentences:
    temp = []
    temp2 = []
    for word, ef, lf in s:
        temp.append(word)
        temp2.append(ef)
    train_words.append(temp)
    train_efs.append(temp2)

valid_sentences = dp.read_origin_file("data/" + args.dataset + "/valid.txt")
valid_words = []
valid_efs = []
for s in valid_sentences:
    temp = []
    temp2 = []
    for word, ef, lf in s:
        temp.append(word)
        temp2.append(ef)
    valid_words.append(temp)
    valid_efs.append(temp2)

test_sentences = dp.read_origin_file("data/" + args.dataset + "/test.txt")
test_words = []
test_efs = []
for s in test_sentences:
    temp = []
    temp2 = []
    for word, ef, lf in s:
        temp.append(word)
        temp2.append(ef)
    test_words.append(temp)
    test_efs.append(temp2)
for e in range(1, args.epochs+1):
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=16'
    print("Epoch: {}".format(e))
    bar.start()
    risks = []
    prisks = []
    nrisks = []
    for step, (x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch, flag_batch) in enumerate(
            mutils.iterateSet(trainSet, batchSize=args.batch_size, mode="TRAIN")):
        bar.update(step)
        batch = [x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch, flag_batch]
        acc, risk, prisk, nrisk = trainer.train_mini_batch(batch, args)
        risks.append(risk)
        prisks.append(prisk)
        nrisks.append(nrisk)
    meanRisk = np.mean(np.array(risks))
    meanRisk2 = np.mean(np.array(prisks))
    meanRisk3 = np.mean(np.array(nrisks))
    print("risk: {}, prisk: {}, nrisk: {}".format(meanRisk, meanRisk2, meanRisk3))

    if e % args.epochs <= 0.8:
        trainer.decay_learning_rate(e, args.lr)
    # train set
    if e != 0:
        pred_train = []
        corr_train = []
        for step, (x_word_train_batch, x_case_train_batch, x_char_train_batch, x_feature_train_batch,
                    y_train_batch) in enumerate(
            mutils.iterateSet(trainSet, batchSize=100, mode="TEST", shuffle=False)):
            trainBatch = [x_word_train_batch, x_case_train_batch, x_char_train_batch, x_feature_train_batch]
            correcLabels = []
            for x in y_train_batch:
                for xi in x:
                    correcLabels.append(xi)
            lengths = [len(x) for x in x_word_train_batch]
            predLabels, _ = trainer.test(trainBatch, lengths)
            correcLabels = np.array(correcLabels)
            # print(predLabels)
            # print(correcLabels)
            assert len(predLabels) == len(correcLabels), "Length x and y label is the same"

            start = 0
            for i, l in enumerate(lengths):
                end = start + l
                p = predLabels[start:end]
                c = correcLabels[start:end]
                pred_train.append(p)
                corr_train.append(c)
                start = end

            # prec, rec, f1 = dp.evaluate_ner_tagging("data/" + args.dataset + "/train.txt",pred_train)

        newSentences = []
        for i, s in enumerate(train_words):
            sent = []
            assert len(s) == len(train_efs[i]) == len(pred_train[i])
            for j, item in enumerate(s):
                sent.append([item, train_efs[i][j], pred_train[i][j]])
            newSentences.append(sent)

        newSentences_, newLabels, newPreds = dp.wordLevelGeneration(newSentences)
        p_train, r_train, f1_train = dp.compute_precision_recall_f1(newLabels, newPreds, args.flag, 1)
        print("Precision: {}, Recall: {}, F1: {}".format(p_train, r_train, f1_train))

        # valid set
        pred_valid = []
        corr_valid = []
        for step, (
                x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch,
                y_test_batch) in enumerate(
            mutils.iterateSet(validSet, batchSize=100, mode="TEST", shuffle=False)):
            validBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch]
            correcLabels = []
            for x in y_test_batch:
                for xi in x:
                    correcLabels.append(xi)
            lengths = [len(x) for x in x_word_test_batch]
            predLabels, _ = trainer.test(validBatch, lengths)
            correcLabels = np.array(correcLabels)
            assert len(predLabels) == len(correcLabels)

            start = 0
            for i, l in enumerate(lengths):
                end = start + l
                p = predLabels[start:end]
                c = correcLabels[start:end]
                pred_valid.append(p)
                corr_valid.append(c)
                start = end

        newSentencesValid = []
        for i, s in enumerate(valid_words):
            sent = []
            assert len(s) == len(valid_efs[i]) == len(pred_valid[i])
            for j, item in enumerate(s):
                sent.append([item, valid_efs[i][j], pred_valid[i][j]])
            newSentencesValid.append(sent)

        newSentencesValid_, newLabelsValid, newPredsValid = dp.wordLevelGeneration(newSentencesValid)
        p_valid, r_valid, f1_valid = dp.compute_precision_recall_f1(newLabelsValid, newPredsValid, args.flag,
                                                                    1)
        print("Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid))

        model_pth = ("saved_model/{}_{}_{}_lr_{}_prior_{}_beta_{}_gamma_{}_percent_{}.pth").format(args.type, args.dataset,
                                                                                            args.flag,
                                                                                            trainer.learningRate,
                                                                                            trainer.m,
                                                                                            trainer.beta,
                                                                                            trainer.gamma,
                                                                                            args.pert)
        
        if f1_valid <= trainer.bestResult:
            time += 1
        else:
            trainer.bestResult = f1_valid
            time = 0
            trainer.save(model_pth)
            print("Saved")
        if time > 5:
            print(("BEST RESULT ON VALIDATE DATA:{}").format(trainer.bestResult))
            break
        if e % 2 == 0:
            pulstmcnn.load_state_dict(
                torch.load(
                    "saved_model/{}_{}_{}_lr_{}_prior_{}_beta_{}_gamma_{}_percent_{}.pth".format(args.type, args.dataset, args.flag,
                                                                                            trainer.learningRate,
                                                                                            trainer.m,
                                                                                            trainer.beta,
                                                                                            trainer.gamma, args.pert)))

            pred_test = []
            corr_test = []
            for step, (
                    x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch,
                    y_test_batch) in enumerate(
                mutils.iterateSet(testSet, batchSize=100, mode="TEST", shuffle=False)):
                testBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch]
                correcLabels = []
                for x in y_test_batch:
                    for xi in x:
                        correcLabels.append(xi)
                lengths = [len(x) for x in x_word_test_batch]
                predLabels, _ = trainer.test(testBatch, lengths)
                correcLabels = np.array(correcLabels)
                assert len(predLabels) == len(correcLabels)

                start = 0
                for i, l in enumerate(lengths):
                    end = start + l
                    p = predLabels[start:end]
                    c = correcLabels[start:end]
                    pred_test.append(p)
                    corr_test.append(c)
                    start = end

            newSentencesTest = []
            for i, s in enumerate(test_words):
                sent = []
                assert len(s) == len(test_efs[i]) == len(pred_test[i])
                for j, item in enumerate(s):
                    sent.append([item, test_efs[i][j], pred_test[i][j]])
                newSentencesTest.append(sent)

            newSentencesValid_, newLabelsValid, newPredsValid = dp.wordLevelGeneration(newSentencesTest)
            p_valid, r_valid, f1_valid = dp.compute_precision_recall_f1(newLabelsValid, newPredsValid, args.flag,
                                                                        1)
            print("Test Result: Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid))
