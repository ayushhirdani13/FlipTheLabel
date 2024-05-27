import os
from time import time
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import model
import data_utils
from loss import flip_loss, truncated_loss
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NCF with Flipped, Normal or Truncated Loss."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset used for training, options: amazon_book, yelp, adressa, movielens",
        default="movielens",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model used for training. options: GMF, NeuMF-end",
        default="NeuMF-end",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="mode used for training. options: {flip, truncated, normal}",
        choices=["flip", "truncated", "normal"],
        default="flip",
    )
    parser.add_argument(
        "--flip_e1", type=int, default=1, help="epoch to start flipping"
    )
    parser.add_argument("--flip_e2", type=int, default=1, help="epoch to stop flipping")
    parser.add_argument("--drop_rate", type=float, help="drop rate", default=0.2)
    parser.add_argument(
        "--num_gradual",
        type=int,
        default=30000,
        help="how many epochs to linearly increase drop_rate",
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=1,
        help="exponent of the drop rate {0.5, 1, 2}",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--epochs", type=int, default=10, help="training epoches")
    parser.add_argument("--eval_freq", type=int, default=2000, help="the freq of eval")
    parser.add_argument(
        "--top_k", type=int, nargs=2, default=[3, 20], help="compute metrics@top_k"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=2048,
        help="batch size for testing and evaluating",
    )
    parser.add_argument(
        "--factor_num",
        type=int,
        default=32,
        help="predictive factors numbers in the model",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="number of layers in MLP model"
    )
    parser.add_argument(
        "--num_ng", type=int, default=1, help="sample negative items for training"
    )
    parser.add_argument("--out", default=True, help="save model or not")
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")

    return parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(2024 + worker_id)


def drop_rate_schedule(iteration):

    drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
    if iteration < args.num_gradual:
        return drop_rate[iteration]
    else:
        return args.drop_rate

########################### Test #####################################
def test(model, test_data_pos, user_pos):
    top_k = args.top_k
    model.eval()
    precision, recall, NDCG, MRR = evaluate.test_all_users(
        model, args.test_batch_size, item_num, test_data_pos, user_pos, top_k
    )

    print(f"################### TEST ######################")
    print(f"Recall@{top_k[0]}: {recall[0]:.4f} Recall@{top_k[1]}:{recall[1]:.4f}")
    print(f"NDCG@{top_k[0]}: {NDCG[0]:.4f} NDCG@{top_k[1]}:{NDCG[1]:.4f}")
    print(
        f"Precision@{top_k[0]}: {precision[0]:.4f} Precision@{top_k[1]}:{precision[1]:.4f}"
    )
    print(f"MRR@{top_k[0]}: {MRR[0]:.4f} MRR@{top_k[1]}:{MRR[1]:.4f}")
    print("################### TEST END ######################")

    return recall[0]

########################### Eval #####################################
def evalModel(model, valid_loader, best_loss, best_recall, count):

    model.eval()
    epoch_loss = 0
    valid_loader.dataset.ng_sample()  # negative sampling
    for user, item, label, noisy_or_not, flips, idx in valid_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        flips = flips.float().cuda()

        prediction = model(user, item)
        if args.mode == 'flip':
            loss, _ = flip_loss(y=prediction, label=label, flips=flips, drop_rate=drop_rate_schedule(count))
        elif args.mode == 'truncated':
            loss = truncated_loss(y=prediction, label=label, drop_rate=drop_rate_schedule(count))
        else:
            loss = F.binary_cross_entropy_with_logits(prediction, label)
        epoch_loss += loss.detach()
    print("################### EVAL ######################")
    print("Eval loss:{}".format(epoch_loss))
    epoch_recall = test(model, test_data_pos, user_pos)

    if epoch_recall > best_recall:
        best_recall = epoch_recall
        if args.out:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(
                model,
                "{}{}-{}_{}-{}.pth".format(
                    model_path, args.model, args.mode, args.drop_rate, args.num_gradual
                ),
            )
    return best_loss, best_recall




if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    torch.manual_seed(2024)  # cpu
    torch.cuda.manual_seed(2024)  # gpu
    np.random.seed(2024)  # numpy
    random.seed(2024)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    data_path = "data/{}/".format(args.dataset)
    model_path = "models/{}/".format(args.dataset)
    print("arguments: %s " % (args))
    print("config model", args.model)
    print("config data path", data_path)
    print("config model path", model_path)

    ############################## PREPARE DATASET ##########################
    (
        train_data,
        valid_data,
        test_data_pos,
        user_pos,
        user_num,
        item_num,
        train_mat,
        train_data_noisy,
    ) = data_utils.load_all(args.dataset, data_path)

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy
    )
    valid_dataset = data_utils.NCFData(valid_data, item_num, train_mat, args.num_ng, 1)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    print(
        "data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(
            user_num, item_num, len(train_data), len(test_data_pos)
        )
    )

    ########################### CREATE MODEL #################################
    if args.model == "NeuMF-pre":  # pre-training. Not used in our work.
        GMF_model_path = model_path + "GMF.pth"
        MLP_model_path = model_path + "MLP.pth"
        NeuMF_model_path = model_path + "NeuMF.pth"
        assert os.path.exists(GMF_model_path), "lack of GMF model"
        assert os.path.exists(MLP_model_path), "lack of MLP model"
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = model.NCF(
        user_num,
        item_num,
        args.factor_num,
        args.num_layers,
        args.dropout,
        args.model,
        GMF_model,
        MLP_model,
    )

    model.cuda()

    if args.model == "NeuMF-pre":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ###################### TRAINING ##########################

    best_loss, best_recall = 1e9, 0
    count = 0
    for epoch in range(args.epochs):
        model.train()
        train_loader.dataset.ng_sample()  # negative sampling
        if args.mode == "flip" and epoch == args.flip_e1:
            print("############ Flip starts ############")
        if args.mode == "flip" and epoch == args.flip_e2+1:
            print("############ Flip ends ############")
        for user, item, label, noisy_or_not, flip, idx in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            # noisy_or_not = noisy_or_not.float().cuda()
            flip = flip.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            if args.mode == "flip":
                if args.flip_e1 <= epoch <= args.flip_e2:
                    loss, flip_inds = flip_loss(
                        y=prediction,
                        label=label,
                        flips=flip,
                        drop_rate=drop_rate_schedule(count),
                    )
                    flip_inds = idx[flip_inds].tolist()
                    train_loader.dataset.flip_labels(flip_inds)
                else:
                    loss = F.binary_cross_entropy_with_logits(prediction, flip)
            elif args.mode == "truncated":
                loss = truncated_loss(
                    y=prediction, label=label, drop_rate=drop_rate_schedule(count)
                )
            else:
                loss = F.binary_cross_entropy_with_logits(prediction, label)

            loss.backward()
            optimizer.step()

            count += 1
        print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
        best_loss, best_recall = evalModel(model, valid_loader, best_loss, best_recall, count)
        model.train()

    print("############################## Training End. ##############################")

    print("Best recall:{}".format(best_recall))
    print("Best recall model saved at : {}".format(model_path))