import torch.nn as nn
import torch
import torch.nn.functional as F


class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.LeakyReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.linear(x)


class HyperGraphBasicConvolution(nn.Module):
    def __init__(self, input_dim):
        super(HyperGraphBasicConvolution, self).__init__()
        self.aggregation = nn.Linear(2 * input_dim, input_dim)

    def forward(self, user_emb, item_emb, group_emb, user_hyper_graph, item_hyper_graph, full_hyper):
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)
        msg = self.aggregation(torch.cat([user_msg, item_msg], dim=1))
        norm_emb = torch.mm(full_hyper, msg)

        return norm_emb, msg


class HyperGraphConvolution(nn.Module):
    """Hyper-graph Convolution for Member-level hyper-graph"""

    def __init__(self, user_hyper_graph, item_hyper_graph, full_hyper, layers,
                 input_dim, device):
        super(HyperGraphConvolution, self).__init__()
        self.layers = layers
        self.user_hyper, self.item_hyper, self.full_hyper_graph = user_hyper_graph, item_hyper_graph, full_hyper
        self.hgnns = [HyperGraphBasicConvolution(input_dim).to(device) for _ in range(layers)]

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        final_ui = [torch.cat([user_emb, item_emb], dim=0)]
        final_g = [group_emb]
        for i in range(len(self.hgnns)):
            hgnn = self.hgnns[i]
            emb, he_msg = hgnn(user_emb, item_emb, group_emb, self.user_hyper, self.item_hyper, self.full_hyper_graph)
            user_emb, item_emb = torch.split(emb, [num_users, num_items])
            final_ui.append(emb)
            final_g.append(he_msg)

        final_ui = torch.sum(torch.stack(final_ui), dim=0)
        final_g = torch.sum(torch.stack(final_g), dim=0)
        return final_ui, final_g


class AlignGroup(nn.Module):
    """AlignGroup"""
    def __init__(self, num_users, num_items, num_groups, args, user_hyper_graph, item_hyper_graph,
                 full_hyper, overlap_graph, device, cl_info, temp):
        super(AlignGroup, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        # Hyper-parameters
        self.emb_dim = args.emb_dim
        self.layers = args.layers
        self.device = args.device
        self.predictor_type = args.predictor

        self.temp = temp
        self.cl_weight = cl_info

        self.overlap_graph = overlap_graph
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)

        # Embedding init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        # Hyper-graph Convolution
        self.hyper_graph_conv = HyperGraphConvolution(user_hyper_graph, item_hyper_graph, full_hyper, self.layers,
                                                      self.emb_dim, device)

        # Prediction Layer
        self.predict = PredictLayer(self.emb_dim)

    def forward(self, group_inputs, user_inputs, pos_item_inputs, neg_item_inputs, members, mode):
        if (group_inputs is not None) and (user_inputs is None):
            return self.group_forward(group_inputs, pos_item_inputs, neg_item_inputs, members, mode)
        else:
            return self.user_forward(user_inputs, pos_item_inputs, neg_item_inputs, members, mode)

    def group_forward(self, group_inputs, pos_item_inputs, neg_item_inputs, members, mode):
        # Group-level graph computation
        group_emb = torch.mm(self.overlap_graph, self.group_embedding.weight)
        # group_emb = self.group_embedding.weight
        # Member-level graph computation
        ui_emb, g_emb = self.hyper_graph_conv(self.user_embedding.weight, self.item_embedding.weight,
                                               group_emb, self.num_users, self.num_items)
        u_emb, i_emb = torch.split(ui_emb, [self.num_users, self.num_items])

        i_emb_pos = i_emb[pos_item_inputs]
        i_emb_neg = i_emb[neg_item_inputs]
        g_emb = g_emb[group_inputs]
        if mode == 'eval':
            loss, pos_prediction = self.BPR_loss(g_emb, i_emb_pos, i_emb_neg)
            return loss, pos_prediction
        else:
            centers = self.get_centers(members, u_emb)
            # cl
            cl_loss = self.InfoNCE(centers, g_emb, self.temp)
            bpr_loss, pos_prediction = self.BPR_loss(g_emb, i_emb_pos, i_emb_neg)
            loss = bpr_loss + cl_loss * self.cl_weight
            return loss, pos_prediction

    def get_centers(self, members, u_emb):
        centers = torch.empty(0).to(self.device)
        for member in members:
            embedding_member = torch.index_select(u_emb, 0, member)
            center = self.geometric_group(embedding_member)
            centers = torch.cat((centers, torch.unsqueeze(center, dim=0)), dim=0)
        return centers

    def geometric_group(self, embedding_member):
        """Geometric bounding and projection for group representation"""
        u_max = torch.max(embedding_member, dim=0).values
        u_min = torch.min(embedding_member, dim=0).values
        center = (u_max + u_min) / 2
        return center

    def BPR_loss(self, g_emb, i_emb_pos, i_emb_neg):
        # For CAMRa2011, we use DOT mode to avoid the dead ReLU
        if self.predictor_type == "MLP":
            pos_prediction = torch.sigmoid(self.predict(g_emb * i_emb_pos))
            neg_prediction = torch.sigmoid(self.predict(g_emb * i_emb_neg))
        else:
            pos_prediction = torch.sum(g_emb * i_emb_pos, dim=-1)
            neg_prediction = torch.sum(g_emb * i_emb_neg, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        return loss, pos_prediction

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def user_forward(self, user_inputs, pos_item_inputs, neg_item_inputs, members, mode):
        u_emb = self.user_embedding(user_inputs)
        i_emb_pos = self.item_embedding(pos_item_inputs)
        i_emb_neg = self.item_embedding(neg_item_inputs)
        if self.predictor_type == "MLP":
            pos_prediction = torch.sigmoid(self.predict(u_emb * i_emb_pos))
            neg_prediction = torch.sigmoid(self.predict(u_emb * i_emb_neg))
        else:
            pos_prediction = torch.sum(u_emb * i_emb_pos, dim=-1)
            neg_prediction = torch.sum(u_emb * i_emb_neg, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        return loss, pos_prediction
