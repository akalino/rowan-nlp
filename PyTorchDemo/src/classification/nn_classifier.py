import argparse
import numpy as np
import random
import torch
import torch.nn as nn

from sklearn.datasets import fetch_20newsgroups
from torch.optim import Adam
from tqdm import tqdm

from compare_features import load_data, build_tfidf_features, build_w2v_features
from models import SimpleNet, SimpleDropNet


def binary_acc(_pred, _target):
    rounded_pred = torch.round(_pred)
    correct = (rounded_pred == _target).float()
    acc = correct.sum() / len(correct)
    return acc


def train_tfidf():

    # define global parameters
    FEATURE_DIM = 300
    EPOCHS = 30
    DROPOUT = 0.2

    train_ds = load_data()
    feats, cv, tft = build_tfidf_features(train_ds)
    # print(feats.shape)

    # convert features to a tensor
    feats_tensor = torch.tensor(feats.todense()).float()
    # print(feats_tensor[0])

    # also convert targets to a tensor
    target_tensor = torch.tensor(train_ds.target).float().unsqueeze(1)
    # print(target_tensor.shape)
    # print(target_tensor[0])

    mod = SimpleNet(feats_tensor.shape[1])
    #mod = SimpleDropNet(feats_tensor.shape[1], DROPOUT)
    opt = Adam(mod.parameters())
    criteria = nn.BCELoss()

    # optional: if there is a GPU available, the model, optimizer and data need to move to the device
    # mod = mod.to(device)
    # criteria = loss.to(device)

    # begin a training loop
    epoch_loss = 0
    epoch_acc = 0
    mod.train()  # need to put the model into training mode
    for j in tqdm(range(EPOCHS)):
        opt.zero_grad()  # reset the gradients each pass
        # forward pass
        preds = mod(feats_tensor)
        # evaluation of forward pass
        current_loss = criteria(preds, target_tensor)
        current_acc = binary_acc(preds, target_tensor)
        # losses are pushed back through the model for an update
        current_loss.backward()
        # optimizer takes a step in the direction of the gradients
        opt.step()
        epoch_loss += current_loss.item()
        epoch_acc += current_acc.item()
    print('Averaged loss: {}'.format(epoch_loss / EPOCHS))
    print('Averaged accuracy: {}'.format(epoch_acc / EPOCHS))
    return mod, cv, tft


def train_w2v():

    # define global parameters
    FEATURE_DIM = 300
    EPOCHS = 50
    DROPOUT = 0.2

    train_ds = load_data()
    wv_feats, w2v = build_w2v_features(train_ds, FEATURE_DIM)
    # print(feats.shape)

    # convert features to a tensor
    feats_tensor = torch.tensor(wv_feats).float()
    # print(feats_tensor[0])

    # also convert targets to a tensor
    target_tensor = torch.tensor(train_ds.target).float().unsqueeze(1)
    # print(target_tensor.shape)
    # print(target_tensor[0])

    #mod = SimpleNet(feats_tensor.shape[1])
    mod = SimpleDropNet(feats_tensor.shape[1], DROPOUT)
    opt = Adam(mod.parameters())
    criteria = nn.BCELoss()

    # optional: if there is a GPU available, the model, optimizer and data need to move to the device
    # mod = mod.to(device)
    # criteria = loss.to(device)

    # begin a training loop
    epoch_loss = 0
    epoch_acc = 0
    mod.train()  # need to put the model into training mode
    for j in tqdm(range(EPOCHS)):
        opt.zero_grad()  # reset the gradients each pass
        # forward pass
        preds = mod(feats_tensor)
        # evaluation of forward pass
        current_loss = criteria(preds, target_tensor)
        current_acc = binary_acc(preds, target_tensor)
        # losses are pushed back through the model for an update
        current_loss.backward()
        # optimizer takes a step in the direction of the gradients
        opt.step()
        epoch_loss += current_loss.item()
        epoch_acc += current_acc.item()
    print('Averaged loss: {}'.format(epoch_loss / EPOCHS))
    print('Averaged accuracy: {}'.format(epoch_acc / EPOCHS))
    return mod, w2v


def evaluate_tfidf(_clf, _cv, _tft):
    categories = ['comp.graphics', 'sci.med']
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    test_cv = _cv.transform(twenty_test.data)
    test_tfidf = _tft.transform(test_cv)
    feats_tensor = torch.tensor(test_tfidf.todense()).float()
    target_tensor = torch.tensor(twenty_test.target).float().unsqueeze(1)
    _clf.eval()
    preds = _clf(feats_tensor)
    rounded_pred = torch.round(preds)
    correct = (rounded_pred == target_tensor).float()
    acc = correct.sum() / len(correct)
    print('Testing accuracy is {}'.format(acc))


def evaluate_w2v(_clf, _w2v):
    _in_dim = 300
    categories = ['comp.graphics', 'sci.med']
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    sentences = []
    features = []
    categories = ['comp.graphics', 'sci.med']
    _test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    for doc in _test.data:
        sentences.append(doc.split(' '))
    for doc in tqdm(sentences):
        doc_vec = np.zeros(_in_dim)
        for tok in doc:
            try:
                doc_vec += _w2v.wv[tok]
            except KeyError:
                pass
        features.append(doc_vec)
    features = np.asarray(features)

    feats_tensor = torch.tensor(features).float()
    target_tensor = torch.tensor(twenty_test.target).float().unsqueeze(1)
    _clf.eval()
    preds = _clf(feats_tensor)
    rounded_pred = torch.round(preds)
    correct = (rounded_pred == target_tensor).float()
    acc = correct.sum() / len(correct)
    print('Testing accuracy is {}'.format(acc))


if __name__ == "__main__":
    # Use argparse to select which model
    parser = argparse.ArgumentParser(prog="nn_classifier",
                                     description="News story classification using two different feature spaces")
    parser.add_argument('-t', '--feature-type', nargs='?', type=str, required=True,
                        help='The type features to use in neural network classification',
                        dest='mt')
    args = parser.parse_args()
    # Let's set our random seeds
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    if args.mt == 'tfidf':
        model, f1, f2 = train_tfidf()
        evaluate_tfidf(model, f1, f2)
    elif args.mt == 'w2v':
        model, wvs = train_w2v()
        evaluate_w2v(model, wvs)
    else:
        print('Did not find a feature set for selection {}, try again'.format(args.mt))
