# BAGCN # A+B+C


import torch
import torch.nn
from helper import *


class BAGCN(torch.nn.Module):


    sub_ent_dim = 0  #
    obj_ent_dim = 1  #

    ent_dim = 0  #
    rel_dim = 0

    def __init__(self, in_channels, out_channels, activation= lambda x:x, dropout=None, residual=None, bias=None):
        super().__init__()

        """

        Parameters
        ----------
        in_channels
        out_channels
        num_heads
        concat
        activation
        dropout
        residual
        bias
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self.w_ent_sub = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.w_ent_obj = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.w_rel = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)

        self.leakyReLU = nn.LeakyReLU(0.2)

        self.a = get_param((1, self.in_channels))
        # self.b = get_param((1, self.in_channels))

        self.kernel_in = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.kernel_out = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.kernel_rel = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        # self.bn_ent = torch.nn.BatchNorm1d(out_channels)
        # self.bn_rel = torch.nn.BatchNorm1d(out_channels)

        self.residual_proj_ent = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.residual_proj_rel = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

        if bias:
            self.bias_ent = get_param((1, self.out_channels))
            self.bias_rel = get_param((1, self.out_channels))



    def forward(self, ent_embed, rel_embed, edge_index, edge_type):


        num_edges = edge_index.size(1) // 2  # E
        num_ent = ent_embed.size(0)  # N
        num_rel = rel_embed.size(0) // 2

        # Step 1: Linear Projection + regularization
        # shape = (N, FIN)
        sub_ent_embed_proj = self.w_ent_sub(ent_embed)
        obj_ent_embed_proj = self.w_ent_obj(ent_embed)
        # shape = (R, FIN)
        rel_embed_proj = self.w_rel(rel_embed)

        # Step 2, 3
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type, out_type = edge_type[:num_edges], edge_type[num_edges:]

        # shape = (N, FOUT) shape = (R, FOUT)
        in_ent_embed,  in_rel_embed= self.aggregate_neighbors(sub_ent_embed_proj, obj_ent_embed_proj, rel_embed_proj, in_index, in_type, ent_embed, num_ent, num_rel, mode='in')
        out_ent_embed, out_rel_embed = self.aggregate_neighbors(sub_ent_embed_proj, obj_ent_embed_proj, rel_embed_proj, out_index, out_type, ent_embed, num_ent, num_rel, mode='out')

        update_ent_embed = in_ent_embed + out_ent_embed

        update_rel_embed = torch.cat([in_rel_embed, out_rel_embed], dim=0)

        # Step 4: Residual/skip connections, bias
        # shape = (N, FOUT)
        if self.residual:
            update_ent_embed = update_ent_embed + self.residual_proj_ent(ent_embed)

            update_rel_embed = update_rel_embed + self.residual_proj_rel(rel_embed)

        if self.bias_ent is not None:
            update_ent_embed += self.bias_ent

        if self.bias_rel is not None:
            update_rel_embed += self.bias_rel

        # update_ent_embed = self.bn_ent(update_ent_embed)
        # update_rel_embed = self.bn_rel(update_rel_embed)

        if self.activation is None:
            return update_ent_embed, update_rel_embed
        else:
            return self.activation(update_ent_embed), self.activation(update_rel_embed)




    # def aggregate_entities(self, sub_ent_embed_proj, obj_ent_embed_proj, edge_index, edge_type, ent_embed, num_rel):
    #     b = getattr(self, 'b')
    #
    #     sub_ent_index = edge_index[self.sub_ent_dim]
    #     obj_ent_index = edge_index[self.obj_ent_dim]
    #
    #     # shape = (E, FIN)
    #     sub_embeding = sub_ent_embed_proj.index_select(self.ent_dim, sub_ent_index)
    #     obj_embeding = obj_ent_embed_proj.index_select(self.ent_dim, obj_ent_index)
    #     so_embeding = sub_embeding + obj_embeding
    #     so_embed = self.leakyReLU(so_embeding)
    #
    #     # shape = (E)
    #     scores_per_edge = (so_embed * b).sum(dim=-1)
    #     # shape = (E, 1)
    #     attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_type, num_rel)
    #     attentions_per_edge = self.dropout(attentions_per_edge)
    #
    #     # shape = (E, FIN) * (E) -> (E, FIN)
    #     attentions_embed = so_embeding * attentions_per_edge
    #
    #     # shape = (R, FIN)
    #     out_rel_embed = self.neighborhood_aggregation(attentions_embed, ent_embed, edge_type, num_rel)
    #
    #     return self.kernel_rel(out_rel_embed)

    def aggregate_neighbors(self, sub_ent_embed_proj, obj_ent_embed_proj, rel_embed_proj, edge_index, edge_type, ent_embed, num_ent, num_rel, mode):

        # Step 2: Edge attention calculation

        kernel = getattr(self, 'kernel_{}'.format(mode))

        a = getattr(self, 'a')
        # b = getattr(self, 'b')
        if mode == 'in':
            edge_type_index = edge_type
        else:
            edge_type_index = edge_type-num_rel

        sub_ent_index = edge_index[self.sub_ent_dim]
        obj_ent_index = edge_index[self.obj_ent_dim]

        # shape = (E, FIN)
        sub_embeding = sub_ent_embed_proj.index_select(self.ent_dim, sub_ent_index)
        obj_embeding = obj_ent_embed_proj.index_select(self.ent_dim, obj_ent_index)
        edge_embedding = rel_embed_proj.index_select(self.rel_dim, edge_type)
        triple_embedding = sub_embeding + obj_embeding + edge_embedding
        # triple_embed = self.leakyReLU(triple_embedding)

        # so_embeding = sub_embeding + obj_embeding
        # so_embed = self.leakyReLU(so_embeding)


        # shape = (E)
        scores_per_edge = self.leakyReLU((triple_embedding * a).sum(dim=-1))
        # scores_per_edge_r = (triple_embed * b).sum(dim=-1)

        # shape = (E, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, sub_ent_index, num_ent)
        attentions_per_edge = self.dropout(attentions_per_edge)

        attentions_per_edge_r = self.neighborhood_aware_softmax(scores_per_edge, edge_type_index, num_rel)
        attentions_per_edge_r = self.dropout(attentions_per_edge_r)



        # Step 3: Neighborhood aggregation
        # shape = (E, FIN) * (E) -> (E, FIN)
        attentions_embed = triple_embedding * attentions_per_edge
        attentions_embed_r = triple_embedding * attentions_per_edge_r

        # shape = (N, FIN)
        out_ent_embed = self.neighborhood_aggregation(attentions_embed, ent_embed, sub_ent_index, num_ent)

        # shape = (R, FIN)
        out_rel_embed = self.neighborhood_aggregation(attentions_embed_r, ent_embed, edge_type_index, num_rel)

        # shape = (E, FIN) * (FIN, FOUT) = (E, FOUT) # shape = (R, FIN) * (FIN, FOUT) = (R, FOUT)
        return kernel(out_ent_embed), self.kernel_rel(out_rel_embed)



    def neighborhood_aware_softmax(self, scores_per_edge, sub_ent_index, num_ent):

        # Calculate the numerator
        # (E)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, sub_ent_index, num_ent)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E) -> (E, 1)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, sub_ent_index, num_ent):
        # shape = (E)
        sub_ent_index_broadcasted = self.explicit_broadcast(sub_ent_index, exp_scores_per_edge)

        # shape = (N)
        size = list(exp_scores_per_edge.shape)
        size[self.ent_dim] = num_ent
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.ent_dim, sub_ent_index_broadcasted, exp_scores_per_edge)

        # shape = (N) -> (E)
        return neighborhood_sums.index_select(self.ent_dim, sub_ent_index)

    def neighborhood_aggregation(self, attentions_embed, ent_embed, sub_ent_index, num_ent):
        size = list(attentions_embed.shape)
        size[self.ent_dim] = num_ent # num_ent
        out_ent_embed = torch.zeros(size, dtype=ent_embed.dtype, device=ent_embed.device)

        # shape = (E) -> (E, FOUT)
        sub_ent_index_broadcasted = self.explicit_broadcast(sub_ent_index, attentions_embed)

        # shape = (E, FOUT) -> (N, FOUT)
        out_ent_embed.scatter_add_(self.ent_dim, sub_ent_index_broadcasted, attentions_embed)

        return out_ent_embed

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)










