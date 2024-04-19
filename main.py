import sys

import torch
import random
import torch.optim as optim
import numpy as np
from metrics import evaluate
from model import AlignGroup
from datetime import datetime
from utils import get_local_time
import argparse
import time
from dataloader import GroupDataset
# from tensorboardX import SummaryWriter
import os
import logging
from sklearn import manifold

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def training(train_loader, epoch, type_m="group", group_member_dict=None):
    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    for batch_id, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(running_device), pi_ni[:, 1].to(running_device)

        optimizer.zero_grad()
        if type_m == 'user':
            loss, _ = train_model(None, user_input, pos_items_input, neg_items_input, None, 'train')
        else:
            members = [torch.LongTensor(group_member_dict[group_id]).to(running_device) for group_id in list(u.cpu().numpy())]
            loss, _ = train_model(user_input, None, pos_items_input, neg_items_input, members, 'train')

        losses.append(loss)
        loss.backward()
        optimizer.step()

    string = (f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    logging.info(string)
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011]", default="Mafengwo") # CAMRa2011, Mafengwo
    parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

    parser.add_argument("--layers", type=int, help="# HyperConv & OverlapConv layers", default=4) # 3 is the best
    parser.add_argument("--emb_dim", type=int, help="User/Item/Group embedding dimensions", default=32)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--topK", type=list, default=[1, 5, 10])

    parser.add_argument("--epoch", type=int, default=100, help="# running epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--loss_type", type=str, default="BPR")
    parser.add_argument("--temp", type=list, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--cl_weight", type=list, default=[0.001, 0.01, 0.1])



    args = parser.parse_args()
    set_seed(args.seed)
    logfilename = '{}-{}.log'.format(args.dataset, get_local_time())

    logfilepath = os.path.join('log/', logfilename)

    file_handler = logging.FileHandler(logfilepath, mode='a', encoding='utf8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    ## t-sne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    logging.info('= ' * 20)
    msg = ('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(msg)
    logging.info(args)

    running_device = torch.device(args.device)

    # Load dataset
    user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
    dataset = GroupDataset(user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset)
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    logging.info(" #Users {}, #Items {}, #Groups {}\n".format(num_users, num_items, num_groups))

    user_hg, item_hg, full_hg = dataset.user_hyper_graph.to(running_device), dataset.item_hyper_graph.to(
        running_device), dataset.full_hg.to(running_device)
    overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)

    group_member_dict = dataset.group_member_dict

    # Prepare model
    logging.info('██Dataset: \t' + args.dataset)

    for idx in range(len(args.cl_weight) * len(args.temp)):
        cl_info = args.cl_weight[idx % (len(args.cl_weight))]
        temp = args.temp[idx // (len(args.cl_weight))]
        logging.info(f"Idx = {idx+1} / {len(args.cl_weight) * len(args.temp)}, cl_weight = {cl_info}, temp = {temp}: ")
        # Prepare model
        train_model = AlignGroup(num_users, num_items, num_groups, args, user_hg, item_hg,
                               full_hg, overlap_graph, running_device, cl_info, temp)
        train_model.to(running_device)

        for epoch_id in range(args.epoch):
            train_model.train()
            g_loader = dataset.get_group_dataloader(args.batch_size)

            group_loss = training(g_loader, epoch_id, "group", group_member_dict)

            user_loss = training(dataset.get_user_dataloader(args.batch_size), epoch_id, "user", group_member_dict)

            hits, ndcgs = evaluate(train_model, dataset.group_test_ratings, dataset.group_test_negatives,
                                   running_device,
                                   args.topK, 'group')

            logging.info("[Epoch {}] Group, Hit@{}: {}, NDCG@{}: {}".format(epoch_id, args.topK, hits, args.topK, ndcgs))

            hrs, ngs = evaluate(train_model, dataset.user_test_ratings, dataset.user_test_negatives, running_device,
                                args.topK, 'user')

            logging.info("[Epoch {}] User, Hit@{}: {}, NDCG@{}: {}".format(epoch_id, args.topK, hrs, args.topK, ngs))

            # tsne
            # g_rep, u_rep, i_rep = train_model.group_embedding.weight.clone(), train_model.user_embedding.weight.clone(), train_model.item_embedding.weight.clone()
            # g_rep = g_rep.cpu().data.numpy()
            # u_rep = u_rep.cpu().data.numpy()
            # i_rep = i_rep.cpu().data.numpy()
            #
            # g_rep = tsne.fit_transform(g_rep)
            # u_rep = tsne.fit_transform(u_rep)
            # i_rep = tsne.fit_transform(i_rep)
            # np.savetxt('./embd/wo/g/' + str(epoch_id) + '.csv', g_rep, delimiter=',')
            # np.savetxt('./embd/wo/u/' + str(epoch_id) + '.csv', u_rep, delimiter=',')
            # np.savetxt('./embd/wo/i/' + str(epoch_id) + '.csv', i_rep, delimiter=',')
        msg = ('## Finishing Time: {}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logging.info(msg)
        logging.info('= ' * 20)
        logging.info("Done!")
