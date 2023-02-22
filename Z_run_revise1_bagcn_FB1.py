import csv
import sys, os, random, json, uuid, time, argparse, logging, logging.config

import torch
from prettytable import PrettyTable
# import time
# import numpy as np
import scipy.io as io


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
######
#在模型中返回评分最高负样本的pred作为loss的输入
#####

def get_args():
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset,Experiment name,and storing parameters
    parser.add_argument('--data', dest="dataset", default='FB15k-237', help='Dataset to use for the experiment')
    parser.add_argument("--name", default='testrun_' + time.strftime("[%Y-%m-%d__%H-%M-%S]", time.localtime()), help='Name of the experiment')
    parser.add_argument("--store_file", dest="store_file", default='results_' + time.strftime("[%Y-%m-%d__%H-%M-%S]", time.localtime()), help='file for storing the last results')

    # Training parameters

    # parser.add_argument("--gpu", type=str, default='7', help='GPU to use, set -1 for CPU')
    parser.add_argument('--batch', dest="batch_size", default=128, type=int, help='Batch size')
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning Rate')
    parser.add_argument("--weight_decay", dest='weight_decay', type=float, default=0.0, help='weight_decay of optimizer')
    parser.add_argument("--epoch", dest='max_epochs', default=500, type=int, help='Maximum number of epochs')
    parser.add_argument("--num_workers", type=int, default=10, help='Maximum number of workers used in DataLoader')
    parser.add_argument("--seed", dest='seed', default=1314, type=int, help='Seed for randomization')
    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-margin', type=float, default=10.0, help='Margin')
    parser.add_argument("--gcn_model_name", dest='gcn_model_name', default='BAGCN')
    parser.add_argument("--score_func", dest='score_func', default='ConvE')

    # BAGCN parameters
    parser.add_argument("--lbl_smooth", dest='lbl_smooth', default=0.1, type=float, help='Label smoothing for true labels')
    # parser.add_argument("--aggr", dest='aggr', default='corr', type=str, help='aggregating style of neighbor nodes and relations')
    parser.add_argument("--bias", dest='bias', default=False, type=bool, help='bias for updating entity embedding')
    # parser.add_argument("--aggr_drop", dest='aggr_drop', default=0.2, type=float, help='dropout after aggrgating')
    parser.add_argument("--gcn_drop", dest='gcn_drop', default=0.6, type=float, help='dropout of gcn')
    # parser.add_argument("--gcn_layer", dest='gcn_layer', default=1, type=int, help='number of gcn layers')
    # parser.add_argument("--gamma", dest='gamma', default=0.5, type=float, help='self attention')
    parser.add_argument("--embed_dim", dest='embed_dim', type=int, default=200, help='Embedding dimension for entity and relation')
    parser.add_argument("--gcn_dim", dest='gcn_dim', default=200, type=int, help='the out embedding size through gcn layer')
    parser.add_argument("--num_heads", dest='num_heads', default=3, type=int, help='number of gat heads')

    # ConvE parameters
    parser.add_argument("--embed_drop", dest='embed_drop', default=0.0, type=float, help='dropout for reshaping entity embeding')
    parser.add_argument("--feature_drop", dest='feat_drop', default=0.0, type=float, help='dropout of feature map')
    parser.add_argument("--hidden_drop", dest='hid_drop', default=0.0, type=float, help='dropout of hidden layer')
    parser.add_argument("--num_filt", dest='num_filt', default=300, type=int, help='number of filters')
    parser.add_argument("--k_z", dest='k_z', default=5, type=int, help='kernel size')
    parser.add_argument("--k_w", dest='k_w', default=10, type=int, help='width of reshaped embedding')
    parser.add_argument("--k_h", dest='k_h', default=20, type=int, help='height of reshaped embedding')
    parser.add_argument("--neg_sampe_ratio", dest='neg_sampe_ratio', default=40, type=int, help='neg_sampe_ratio')

    # Logging parameters
    parser.add_argument('--logdir', dest="log_dir", default='./log/', help='Log directory')
    parser.add_argument('--config', dest="config_dir", default='./config/', help='Config directory')

    return parser.parse_args()



from dataloader import *
from model.models_bagcn import *

class Runner(object):

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params
        self.logger = get_logger('{}_{}_{}_{}'.format(self.p.dataset, self.p.gcn_model_name, self.p.score_func, self.p.name), self.p.log_dir, self.p.config_dir)


        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if torch.cuda.is_available():
            # set_gpu(self.p.gpu)
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model(self.p.gcn_model_name, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		    The dataloader for different data splits

        """

        self.ent2id = json.load(open('./data/{}/ent2id.json'.format(self.p.dataset),'r'))
        self.rel2id = json.load(open('./data/{}/rel2id.json'.format(self.p.dataset),'r'))

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}


        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2


        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        # for (sub, rel), obj in self.sr2o.items():
        #     self.triples['train'].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})

        for sub, rel, obj in self.data['train']:
            rel_inv = rel + self.p.num_rel  # inverse_rel相反关系
            # sub_samp = len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
            # sub_samp = np.sqrt(1 / sub_samp)
            # print("sub_samp:{}".format(sub_samp))

            self.triples['train'].append(
                {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
            self.triples['train'].append(
                {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        # print(len(self.triples['test_tail']))
        # print(len(self.triples['test_head']))


        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn      = dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)


        edge_index = torch.LongTensor(edge_index).to(self.device).t()  # shape: (2,num_triples*2). first row: head   second row: tail
        edge_type = torch.LongTensor(edge_type).to(self.device)


        return edge_index, edge_type

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split=='train':
            triple, label, neg_tail = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_tail
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def add_model(self, gcn_model_name, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = '{}_{}'.format(gcn_model_name, score_func)

        if model_name.lower() == 'bagcn_distmult':
            model = BAGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'bagcn_conve':
            model = BAGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'bagcn_convd':
            model = BAGCN_ConvD(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model


    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.weight_decay)





    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args'	: vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """

        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val_mrr = state['best_val']['mrr']
        self.best_val = state['best_val']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def store_results(self, results):
        """
        Function to store results

        Parameters
        ----------
        res_table: results

        Returns
        -------
        """

        results_table = PrettyTable([self.p.dataset, 'left/tail', 'right/head', 'mean'])
        mr = ['MR', results['left_mr'], results['right_mr'], results['mr']]
        mrr = ['MRR', results['left_mrr'], results['right_mrr'], results['mrr']]
        hit = []
        for i in range(10):
            hit.append(['Hit@{}'.format(i + 1), results['left_hits@{}'.format(i + 1)], results['right_hits@{}'.format(i + 1)], results['hits@{}'.format(i + 1)]])
        results_table.add_row(mr)
        results_table.add_row(mrr)
        results_table.add_rows(hit)
        self.logger.info(results_table)

        f = open('./results/{}_{}_{}_{}.txt'.format(self.p.dataset, self.p.gcn_model_name, self.p.score_func, self.p.store_file), 'w')
        f.write(str(results_table))
        f.close()

    def run_epoch(self, epoch):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        epoch_loss: The loss value after the completion of one epoch
        """

        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            sub, rel, obj, label, neg_tail= self.read_batch(batch, 'train')
            # print("neg:{}".format(neg_tail))

            pred = self.model.forward(sub, rel, neg_tail, 'train')

            loss = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train Loss:{:.5}, Best_Val MRR:{:.5}, \t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))
        epoch_loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.5}\n'.format(epoch, epoch_loss))
        return epoch_loss

    def predict(self, split, mode):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """

        if split == "valid":
            self.model.eval()
            with torch.no_grad():
                results = {}
                train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
                for step, batch in enumerate(train_iter):
                    sub, rel, obj, label = self.read_batch(batch, split)
                    pred = self.model.forward(sub, rel, None, None)
                    b_range = torch.arange(pred.size()[0], device=self.device)
                    target_pred = pred[b_range, obj]
                    pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                    pred[b_range, obj] = target_pred
                    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                    ranks = ranks.float()
                    results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                    results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                    results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                    for k in range(10):
                        results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits@{}'.format(k + 1), 0.0)
                    if step % 100 == 0:
                        self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
            return results

        else:
            self.model.eval()
            with torch.no_grad():
                results = {}
                train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

                #保存三元组针对所有尾实体的排名情况
                f_rank_all = open('./ranks_index_top10/{}_{}_{}_{}_ranks_{}.csv'.format(self.p.dataset, mode.split('_')[0], self.p.gcn_model_name, self.p.score_func, self.p.name), 'w', encoding="UTF8", newline='')
                writer_all = csv.writer(f_rank_all, delimiter=",")

                #保存三元组针对当前尾实体的排名情况
                f_rank_index = open('./ranks/{}_{}_{}_{}_ranks_{}.txt'.format(self.p.dataset, mode.split('_')[0], self.p.gcn_model_name, self.p.score_func, self.p.name), 'w', encoding="UTF8", newline='')
                writer = csv.writer(f_rank_index, delimiter='\t')

                for step, batch in enumerate(train_iter):
                    sub, rel, obj, label = self.read_batch(batch, split)
                    pred = self.model.forward(sub, rel, None, None)

                    ####针对所有排名 返回排名前10的索引 前三列对应h r t后10列对应排名前10的索引
                    rank_index = torch.argsort(pred, dim=1, descending=True)[:, :10]
                    t_rank_index = torch.cat([sub.unsqueeze(1), rel.unsqueeze(1), obj.unsqueeze(1), rank_index], dim=1)
                    t_rank_index = t_rank_index.cpu().numpy().astype(int)
                    writer_all.writerows(t_rank_index)
                    ######

                    b_range = torch.arange(pred.size()[0], device=self.device)
                    target_pred = pred[b_range, obj]
                    pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                    pred[b_range, obj] = target_pred
                    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                    ranks = ranks.float()

                    #####三元组排名
                    triple_rank = torch.cat([sub.unsqueeze(1), rel.unsqueeze(1), obj.unsqueeze(1), ranks.unsqueeze(1)], dim=1)
                    triple_rank = triple_rank.cpu().numpy().astype(int)
                    writer.writerows(triple_rank)
                    #####

                    results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                    results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                    results['mrr'] = torch.sum(1.0/ranks).item() + results.get('mrr', 0.0)
                    for k in range(10):
                        results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)
                    if step % 100 == 0:
                        self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

            return results

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """

        left_results = self.predict(split=split, mode='tail_batch')  # predict tail according to (head rel)
        right_results = self.predict(split=split, mode='head_batch') # predict head according to (tail rel_inv)
        results = get_combined_results(left_results, right_results)

        self.logger.info('[Epoch {} {}]: Current_MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'],results['right_mrr'],results['mrr']))  # .5代表保留五位有效数字
        return results


    def fit(self):
        """
        Function to run training and evaluation of model
        The whole process of train,valid and test

        Parameters
        ----------

        Returns
        -------
        """

        self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
        save_path = os.path.join('./torch_saved/', '{}_{}_{}_{}'.format(self.p.dataset, self.p.gcn_model_name, self.p.score_func, self.p.name))
        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch(epoch)
            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
                kill_cnt = 0
                self.p.margin = 10
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.margin > 0:
                    self.p.margin -= 1
                    self.logger.info('Margin decay on saturation, updated value of margin: {}'.format(self.p.margin))
                if self.p.margin < 8:
                    print("Early stopping!")
                    break
            self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Best_Valid MRR: {:.5}, \n\n\n'.format(epoch, train_loss, self.best_val_mrr))

        # Restoring model corresponding to the best validation performance and evaluation on test data
        self.logger.info('Loading best model, evaluating on test data')
        self.load_model(save_path)
        test_results = self.evaluate('test', epoch)

        if self.p.dataset == 'FB15k-237' or self.p.dataset == 'WN18RR':
            f = open('./results_nn/nn_{}_{}_{}_{}'.format(self.p.dataset, self.p.gcn_model_name, self.p.score_func,
                                                          self.p.store_file), 'w')
            f.write('all_{}'.format(self.p.dataset) + '\t' + 'tail' + '\t' + 'head' + '\t' + 'mean' + '\n')
            rstt = {}

            for m in ['tail', 'head']:
                f_rank = open('./ranks/{}_{}_{}_{}_ranks_{}.txt'.format(self.p.dataset, m, self.p.gcn_model_name,
                                                                        self.p.score_func, self.p.name),
                              'r').read().split('\n')
                f_rank.remove(f_rank[-1])
                print('len(f_rank):{}'.format(len(f_rank)))
                if m == 'head':
                    mr = 0
                    mrr = 0
                    hit1 = 0
                    hit3 = 0
                    hit10 = 0
                    for line in f_rank:
                        s = line.split('\t')
                        mr += int(s[3])
                        mrr += 1 / int(s[3])
                        if int(s[3]) <= 1:
                            hit1 += 1
                        if int(s[3]) <= 3:
                            hit3 += 1
                        if int(s[3]) <= 10:
                            hit10 += 1
                    mr = mr / len(f_rank)
                    mrr = mrr / len(f_rank)
                    hit1 = hit1 / len(f_rank)
                    hit3 = hit3 / len(f_rank)
                    hit10 = hit10 / len(f_rank)
                    rstt['head_mr'] = mr
                    rstt['head_mrr'] = mrr
                    rstt['head_hit1'] = hit1
                    rstt['head_hit3'] = hit3
                    rstt['head_hit10'] = hit10
                if m == 'tail':
                    mr = 0
                    mrr = 0
                    hit1 = 0
                    hit3 = 0
                    hit10 = 0
                    for line in f_rank:
                        s = line.split('\t')
                        mr += int(s[3])
                        mrr += 1 / int(s[3])
                        if int(s[3]) <= 1:
                            hit1 += 1
                        if int(s[3]) <= 3:
                            hit3 += 1
                        if int(s[3]) <= 10:
                            hit10 += 1
                    mr = mr / len(f_rank)
                    mrr = mrr / len(f_rank)
                    hit1 = hit1 / len(f_rank)
                    hit3 = hit3 / len(f_rank)
                    hit10 = hit10 / len(f_rank)
                    rstt['tail_mr'] = mr
                    rstt['tail_mrr'] = mrr
                    rstt['tail_hit1'] = hit1
                    rstt['tail_hit3'] = hit3
                    rstt['tail_hit10'] = hit10
            rstt['mr'] = (rstt['tail_mr'] + rstt['head_mr']) / 2
            rstt['mrr'] = (rstt['tail_mrr'] + rstt['head_mrr']) / 2
            rstt['hit1'] = (rstt['tail_hit1'] + rstt['head_hit1']) / 2
            rstt['hit3'] = (rstt['tail_hit3'] + rstt['head_hit3']) / 2
            rstt['hit10'] = (rstt['tail_hit10'] + rstt['head_hit10']) / 2

            f.write('mr' + '\t' + str(rstt['tail_mr']) + '\t' + str(rstt['head_mr']) + '\t' + str(rstt['mr']) + '\n')
            f.write(
                'mrr' + '\t' + str(rstt['tail_mrr']) + '\t' + str(rstt['head_mrr']) + '\t' + str(rstt['mrr']) + '\n')
            f.write('hit1' + '\t' + str(rstt['tail_hit1']) + '\t' + str(rstt['head_hit1']) + '\t' + str(
                rstt['hit1']) + '\n')
            f.write('hit3' + '\t' + str(rstt['tail_hit3']) + '\t' + str(rstt['head_hit3']) + '\t' + str(
                rstt['hit3']) + '\n')
            f.write('hit10' + '\t' + str(rstt['tail_hit10']) + '\t' + str(rstt['head_hit10']) + '\t' + str(
                rstt['hit10']) + '\n')
            f.write('\n')

            for tri in ['1-1', '1-n', 'n-1', 'n-n']:
                f.write('{}_{}'.format(tri, self.p.dataset) + '\t' + 'tail' + '\t' + 'head' + '\t' + 'mean' + '\n')
                f_tri = open('./data/{}/{}.txt'.format(self.p.dataset, tri), 'r').read().split('\n')
                f_tri.remove(f_tri[-1])
                rst = {}
                print('len(f_tri):{}_{}'.format(tri, len(f_tri)))
                for m in ['tail', 'head']:
                    f_rank = open('./ranks/{}_{}_{}_{}_ranks_{}.txt'.format(self.p.dataset, m, self.p.gcn_model_name,
                                                                            self.p.score_func, self.p.name),
                                  'r').read().split('\n')
                    f_rank.remove(f_rank[-1])
                    tri_rank = {}

                    for line in f_rank:
                        s = line.split('\t')
                        tri_rank[s[0] + '\t' + s[1] + '\t' + s[2]] = s[3]

                    if m == 'head':
                        mr = 0
                        mrr = 0
                        hit1 = 0
                        hit3 = 0
                        hit10 = 0
                        for line in f_tri:
                            s = line.split('\t')
                            newline = s[2] + '\t' + '{}'.format(str(int(s[1]) + self.p.num_rel)) + '\t' + s[0]
                            mr += int(tri_rank[newline])
                            mrr += (1 / int(tri_rank[newline]))
                            if int(tri_rank[newline]) <= 1:
                                hit1 += 1
                            if int(tri_rank[newline]) <= 3:
                                hit3 += 1
                            if int(tri_rank[newline]) <= 10:
                                hit10 += 1
                        mr = mr / len(f_tri)
                        mrr = mrr / len(f_tri)
                        hit1 = hit1 / len(f_tri)
                        hit3 = hit3 / len(f_tri)
                        hit10 = hit10 / len(f_tri)
                        rst['head_mr'] = mr
                        rst['head_mrr'] = mrr
                        rst['head_hit1'] = hit1
                        rst['head_hit3'] = hit3
                        rst['head_hit10'] = hit10

                    if m == 'tail':
                        mr = 0
                        mrr = 0
                        hit1 = 0
                        hit3 = 0
                        hit10 = 0
                        for line in f_tri:
                            mr += int(tri_rank[line])
                            mrr += (1 / int(tri_rank[line]))
                            if int(tri_rank[line]) <= 1:
                                hit1 += 1
                            if int(tri_rank[line]) <= 3:
                                hit3 += 1
                            if int(tri_rank[line]) <= 10:
                                hit10 += 1
                        mr = mr / len(f_tri)
                        mrr = mrr / len(f_tri)
                        hit1 = hit1 / len(f_tri)
                        hit3 = hit3 / len(f_tri)
                        hit10 = hit10 / len(f_tri)
                        rst['tail_mr'] = mr
                        rst['tail_mrr'] = mrr
                        rst['tail_hit1'] = hit1
                        rst['tail_hit3'] = hit3
                        rst['tail_hit10'] = hit10

                rst['mr'] = (rst['tail_mr'] + rst['head_mr']) / 2
                rst['mrr'] = (rst['tail_mrr'] + rst['head_mrr']) / 2
                rst['hit1'] = (rst['tail_hit1'] + rst['head_hit1']) / 2
                rst['hit3'] = (rst['tail_hit3'] + rst['head_hit3']) / 2
                rst['hit10'] = (rst['tail_hit10'] + rst['head_hit10']) / 2
                f.write('mr' + '\t' + str(rst['tail_mr']) + '\t' + str(rst['head_mr']) + '\t' + str(rst['mr']) + '\n')
                f.write(
                    'mrr' + '\t' + str(rst['tail_mrr']) + '\t' + str(rst['head_mrr']) + '\t' + str(rst['mrr']) + '\n')
                f.write('hit1' + '\t' + str(rst['tail_hit1']) + '\t' + str(rst['head_hit1']) + '\t' + str(
                    rst['hit1']) + '\n')
                f.write('hit3' + '\t' + str(rst['tail_hit3']) + '\t' + str(rst['head_hit3']) + '\t' + str(
                    rst['hit3']) + '\n')
                f.write('hit10' + '\t' + str(rst['tail_hit10']) + '\t' + str(rst['head_hit10']) + '\t' + str(
                    rst['hit10']) + '\n')
                f.write('\n')
            f.close()


        self.store_results(test_results)




if __name__ == "__main__":

    # set_gpu(args.gpu)
    args = get_args()

    # #####
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子（多块GPU）
    # #####

    print("torch.__version__: {}".format(torch.__version__))
    print("torch.version.cuda: {}".format(torch.version.cuda))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))

    model = Runner(args)
    model.fit()



