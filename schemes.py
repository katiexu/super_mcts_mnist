import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from dataset.MNIST import MNISTDataLoaders
from FusionModel import QNet
from FusionModel import translator
from Arguments import Arguments
import random


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    # print("\nTest mae: {}".format(metrics['mae']))
    # print("Test correlation: {}".format(metrics['corr']))
    # print("Test multi-class accuracy: {}".format(metrics['multi_acc']))
    # print("Test binary accuracy: {}".format(metrics['bi_acc']))
    # print("Test f1 score: {}".format(metrics['f1']))
    pass

def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for data_image, target in data_loader:
        data_image = data_image.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()
        output = model(data_image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = []
    output_all = []
    with torch.no_grad():
        for data_image, target in data_loader:
            data_image = data_image.to(args.device)
            target = target.to(args.device)
            output = model(data_image)
            instant_loss = criterion(output, target).item()
            total_loss += instant_loss
            target_all.append(target)
            output_all.append(output)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return total_loss, accuracy


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    target_all = []
    output_all = []
    with torch.no_grad():
        for data_image, target in data_loader:
            data_image, target = data_image.to(args.device), target.to(args.device)
            output = model(data_image)

            target_all.append(target)
            output_all.append(output)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    # print(f"{split} set accuracy: {accuracy}")
    print(f"test set accuracy: {accuracy}\n")

    metrics['acc'] = accuracy

    # metrics['mae'] = np.mean(np.absolute(output - target)).item()
    # metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    # metrics['multi_acc'] = round(sum(np.round(output) == np.round(target)) / float(len(target)), 5).item()
    # true_label = (target >= 0)
    # pred_label = (output >= 0)
    # metrics['bi_acc'] = accuracy_score(true_label, pred_label).item()
    # metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()
    return metrics


def Scheme_no_train(design, weight=None):
    args = Arguments()
    train_loader, val_loader, test_loader = MNISTDataLoaders(args)
    model = QNet(args, design).to(args.device)
    if weight == None:
        model.load_state_dict(torch.load('base_weight_tq'), strict=False)
    else:
        model.load_state_dict(weight, strict=False)
    start = time.time()
    metrics = evaluate(model, test_loader, args)
    end = time.time()
    print("Running time: %s seconds" % (end - start))

    display(metrics)
    report = {'mae': metrics['mae']}

    return report


def Scheme(design, weight=None):
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    args = Arguments()
    # if torch.cuda.is_available() and args.device == 'cuda':
    #     print("using cuda device")
    # else:
    #     print("using cpu device")
    train_loader, val_loader, test_loader = MNISTDataLoaders(args)
    model = QNet(args, design).to(args.device)
    # if weight == None:
    #     # model.load_state_dict(torch.load('classical_weight'), strict= False)
    #     model.load_state_dict(torch.load('base_weight_tq'), strict=False)
    # else:
    #     model.load_state_dict(weight, strict=False)
    # criterion = nn.L1Loss(reduction='sum')
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam([
    #     {'params': model.ClassicalLayer_a.parameters()},
    #     {'params': model.ClassicalLayer_v.parameters()},
    #     {'params': model.ClassicalLayer_t.parameters()},
    #     {'params': model.ProjLayer_a.parameters()},
    #     {'params': model.ProjLayer_v.parameters()},
    #     {'params': model.ProjLayer_t.parameters()},
    #     {'params': model.QuantumLayer.parameters(), 'lr': args.qlr},
    #     {'params': model.Regressor.parameters()}
    #     ], lr=args.clr)
    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_val_loss = 10000

    start = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, args)
        train_loss, train_acc = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss, val_acc = test(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     print(epoch, train_loss, val_loss, 'saving model')
        #     best_model = copy.deepcopy(model)
        # else:
        #     print(epoch, train_loss, val_loss)
        # metrics = evaluate(model, test_loader, args)
        # display(metrics)
        print(f"Epoch: {epoch},", f"train loss: {train_loss},", f"train set accuracy: {train_acc},", f"valid loss: {val_loss},", f"valid set accuracy: {val_acc}")
        # print(epoch, train_loss, train_acc, val_loss, val_acc)
    end = time.time()
    # print("Running time: %s seconds" % (end - start))
    best_model = model
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'acc': metrics['acc']}

    ## store classical weights
    # del best_model.QuantumLayer
    # torch.save(best_model.state_dict(), 'base_weight_tq_2')
    return best_model, report


if __name__ == '__main__':
    # change_code = None
    # change_code = [5, 3, 0, 1, 0, 0]
    # change_code = [6, 1, 1, 2, 1, 0]
    change_code = [1, 0, 3, 2, 2, 3, 0, 1, 2]
    design = translator(change_code)
    best_model, report = Scheme(design)
    # report = Scheme_no_train(design)
    print("end")