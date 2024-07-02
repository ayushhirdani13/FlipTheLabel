from csv import writer
import os
from time import time
import argparse
import random
import numpy as np

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

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
    parser.add_argument(
        "--top_k", type=int, nargs='+', default=[3, 20], help="compute metrics@top_k"
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
    
def get_results_dict(results, top_k):
    results_dict = {}
    for i, k in enumerate(top_k):
        results_dict[f"Recall@{k}"] = results["recall"][i]
        results_dict[f"NDCG@{k}"] = results["NDCG"][i]
        results_dict[f"Precision@{k}"] = results["precision"][i]
        results_dict[f"MRR@{k}"] = results["MRR"][i]
    return results_dict


########################### Test #####################################
def test(model, test_data_pos, user_pos):
    top_k = args.top_k
    model.eval()
    precision, recall, NDCG, MRR = evaluate.test_all_users(
        model, item_num - 1, item_num, test_data_pos, user_pos, top_k
    )

    test_results = {
        "precision": precision,
        "recall": recall,
        "NDCG": NDCG,
        "MRR": MRR,
    }

    test_results_dict = get_results_dict(test_results, top_k)

    print(f"################### TEST ######################")
    for k, v in test_results_dict.items():
        print(f"{k}: {v:.4f}")
    print("################### TEST END ######################")

    return recall[0], test_results


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
        if args.mode == "flip":
            loss_all, _ = flip_loss(
                y=prediction,
                label=label,
                flips=flips,
                drop_rate=drop_rate_schedule(count),
            )
            loss = loss_all.mean()
        elif args.mode == "truncated":
            loss_all = truncated_loss(
                y=prediction, label=label, drop_rate=drop_rate_schedule(count)
            )
            loss = loss_all.mean()
        else:
            loss = F.binary_cross_entropy_with_logits(prediction, label)
        epoch_loss += loss.detach()
    print("################### EVAL ######################")
    print("Eval loss:{}".format(epoch_loss))
    recall, test_results = test(model, test_data_pos, user_pos)

    if recall > best_recall:
        best_recall = recall
        if args.out:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(
                model,
                "{}{}-{}_{}-{}.pth".format(
                    model_path, args.model, args.mode, args.drop_rate, args.num_gradual
                ),
            )
    return best_loss, best_recall, test_results


if __name__ == "__main__":
    start_time = time()

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

    print(
        "############################## Training Start. ##############################"
    )
    writer = SummaryWriter(f"runs/{args.dataset}_{args.model}_{args.mode}")
    print("Initial Advantage Ratio: {:.4f}".format(train_dataset.get_advantage_ratio()))
    best_loss, best_recall = 1e9, 0
    count = 0
    test_results = []
    results_df = pd.DataFrame()
    for epoch in range(args.epochs):
        model.train()
        train_loader.dataset.ng_sample()  # negative sampling
        if args.mode == "flip" and epoch == args.flip_e1:
            print("############ Flip starts ############")
        if args.mode == "flip" and epoch == args.flip_e2 + 1:
            print("############ Flip ends ############")
        for user, item, label, noisy_or_not, flip, idx in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            noisy_or_not = noisy_or_not.float().cuda()
            flip = flip.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            if args.mode == "flip":
                if args.flip_e1 <= epoch <= args.flip_e2:
                    loss_all, flip_inds = flip_loss(
                        y=prediction,
                        label=label,
                        flips=flip,
                        drop_rate=drop_rate_schedule(count),
                    )
                    flip_inds = idx[flip_inds].tolist()
                    train_loader.dataset.flip_labels(flip_inds)
                else:
                    loss_all = F.binary_cross_entropy_with_logits(
                        prediction, flip, reduction="none"
                    )
                loss = loss_all.mean()
            elif args.mode == "truncated":
                loss = truncated_loss(
                    y=prediction, label=label, drop_rate=drop_rate_schedule(count)
                )
                loss_all = F.binary_cross_entropy_with_logits(
                    prediction, label, reduction="none"
                )
            else:
                loss_all = F.binary_cross_entropy_with_logits(
                    prediction, label, reduction="none"
                )
                loss = loss_all.mean()

            true_mask = (noisy_or_not == 1) & (label == 1)
            noisy_mask = (noisy_or_not == 0) & (label == 1)
            neg_mask = label == 0

            tp_loss = loss_all[true_mask].mean()
            fp_loss = loss_all[noisy_mask].mean()
            neg_loss = loss_all[neg_mask].mean()

            writer.add_scalars(
                "Training Losses",
                {
                    "Training Loss": loss,
                    "True Positive Loss": tp_loss,
                    "False Positive Loss": fp_loss,
                    "Negative Loss": neg_loss,
                },
                count,
            )

            loss.backward()
            optimizer.step()

            count += 1
        print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
        best_loss, curr_recall, curr_test_results = evalModel(
            model, valid_loader, best_loss, best_recall, count
        )
        test_results.append(curr_test_results)
        if curr_recall > best_recall:
            best_recall = curr_recall
            best_recall_idx = epoch

        curr_results_dict = get_results_dict(curr_test_results, args.top_k)
        curr_results_df = pd.DataFrame(curr_results_dict, index=[epoch])
        results_df = pd.concat([results_df, curr_results_df])
        model.train()

    writer.flush()
    writer.close()
    print("############################## Training End. ##############################")

    if args.mode == "flip":
        print(
            "Final Advantage Ratio after Flipping: {:.4f}".format(
                train_dataset.get_advantage_ratio()
            )
        )

    print("Results Dataframe:")
    print(results_df)
    if not os.path.exists("results"):
        os.makedirs("results")
    results_df.to_csv(
        f"results/{args.dataset}_{args.model}_{args.mode}_{args.drop_rate}_{args.num_gradual}.csv",
        float_format="%.4f",
    )
    print("#" * 20)
    print("Best recall:{:.4f}".format(best_recall))
    print("Best Recall Epoch:{}".format(best_recall_idx))
    print("Best Recall Metrics:")
    best_results = test_results[best_recall_idx]
    best_results_dict = get_results_dict(best_results, args.top_k)
    for k, v in best_results_dict.items():
        print(f"{k}: {v:.4f}")

    end_time = time()
    print("Total Time: {:.2f}s".format(end_time - start_time))