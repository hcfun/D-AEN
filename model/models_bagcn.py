import torch.nn

from helper import *
from model.BAGCN import BAGCN

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p		= params
        # self.act	= torch.tanh
        self.bceloss	= torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class BAGCNBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(BAGCNBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.ent_embed = get_param((self.p.num_ent, self.p.embed_dim))
        self.rel_embed = get_param((num_rel * 2, self.p.embed_dim))

        self.attenlayers = [BAGCN(self.p.embed_dim, self.p.gcn_dim, activation=torch.tanh, dropout=self.p.gcn_drop, residual=True, bias=True) for _ in range(self.p.num_heads)]
        for i, attention in enumerate(self.attenlayers):
            self.add_module('attention_{}'.format(i), attention)
        self.outattenlayer = BAGCN(self.p.gcn_dim * self.p.num_heads, self.p.gcn_dim, activation=torch.tanh, dropout=self.p.gcn_drop, residual=True, bias=True)



    def forward_base(self, sub, rel, drop):
        e_embed = drop(self.ent_embed)
        r_embed = drop(self.rel_embed)
        embed = [BAGCN(e_embed, r_embed, self.edge_index, self.edge_type) for BAGCN in self.attenlayers]
        # e_embed, r_embed = torch.cat([BAGCN(e_embed, self.rel_embed, self.edge_index, self.edge_type)[0] for BAGCN in self.attenlayers], dim=1), torch.cat([BAGCN(self.e_embed, self.rel_embed, self.edge_index, self.edge_type)[1] for BAGCN in self.attenlayers], dim=1)
        e_embed, r_embed = torch.cat([e[0] for e in embed], dim=1), torch.cat([e[1] for e in embed], dim=1)
        e_embed = drop(e_embed)
        r_embed = drop(r_embed)
        e_embed, r_embed = self.outattenlayer(e_embed, r_embed, self.edge_index, self.edge_type)

        # e_embed = self.act(e_embed)


        sub_emb = torch.index_select(e_embed, 0, sub)
        rel_emb = torch.index_select(r_embed, 0, rel)

        return sub_emb, rel_emb, e_embed


class BAGCN_DistMult(BAGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.gcn_drop = torch.nn.Dropout(self.p.gcn_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, ent_embed= self.forward_base(sub, rel, self.gcn_drop)

        obj_emb = sub_emb * rel_emb
        x = torch.mm(obj_emb, ent_embed.transpose(1, 0))

        return torch.sigmoid(x)



class BAGCN_ConvD(BAGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt_convd)
        self.bn2 = torch.nn.BatchNorm1d(self.p.gcn_dim)

        self.gcn_drop = nn.Dropout(self.p.gcn_drop)

        self.feature_drop = nn.Dropout(self.p.feat_drop_convd)
        self.hidden_drop = nn.Dropout(self.p.hid_drop_convd)

        self.conv = self.conv = nn.Conv2d(1, out_channels=self.p.num_filt_convd, kernel_size=(self.p.kz_h, self.p.kz_w), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = 2 - self.p.kz_h + 1
        flat_sz_w = self.p.gcn_dim - self.p.kz_w + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt_convd

        self.fc = torch.nn.Linear(self.flat_sz, self.p.gcn_dim)

    def forward(self, sub, rel):
        sub_emb, rel_emb, ent_embed= self.forward_base(sub, rel, self.gcn_drop)

        x = torch.cat([sub_emb.unsqueeze(1), rel_emb.unsqueeze(1)], dim=1).unsqueeze(1)
        x = self.bn0(x)
        x = self.conv(x)
        x = self.feature_drop(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, ent_embed.transpose(1, 0))

        return torch.sigmoid(x)



class BAGCN_ConvE(BAGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.gcn_dim)

        self.gcn_drop = nn.Dropout(self.p.gcn_drop)

        self.embed_drop = nn.Dropout(self.p.embed_drop)
        self.feature_drop = nn.Dropout(self.p.feat_drop)
        self.hidden_drop = nn.Dropout(self.p.hid_drop)
        self.conv = nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.k_z, self.p.k_z), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.k_z + 1
        flat_sz_w = self.p.k_h - self.p.k_z + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.gcn_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.gcn_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.gcn_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel, neg_tail, run_mode):
        sub_emb, rel_emb, ent_embed= self.forward_base(sub, rel, self.gcn_drop)

        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.embed_drop(stk_inp)
        x = self.bn0(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, ent_embed.transpose(1, 0))
        x = torch.sigmoid(x)

        if run_mode=='train':
            x = torch.stack([x[i].index_select(0, neg_tail[i]) for i in range(x.shape[0])], 0)

        return x

