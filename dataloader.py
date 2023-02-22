from helper import *
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        trp_label = self.get_label()

        neg_tail = torch.LongTensor(self.negative_sampling(triple, label))

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)
        return triple, trp_label, neg_tail


    @staticmethod
    def collate_fn(data):
        # 这里的data就是getitem打包之后的按batch输出的dataset
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        neg_tail = torch.stack([_[2] for _ in data], dim=0)

        return triple, trp_label, neg_tail

    def negative_sampling(self, triple, label):
        pos_tail = triple[2]
        mask = np.ones([self.p.num_ent], dtype=np.bool)  # one-hot编码的变形
        mask[label] = 0  # 将真实出现的标签位置标记为0,在下一步中，只在为1的位置进行采样，也就是负采样
        neg_tail = np.int32(np.random.choice(self.entities[mask], self.p.neg_sampe_ratio, replace=False)).reshape([-1])  # false：不能取相同的数
        neg_tail = np.concatenate((pos_tail.reshape([-1]), neg_tail))  # 连接真标签和负采样id

        return neg_tail


    def get_label(self):
        # y = np.zeros([self.p.num_ent], dtype=np.float32)
        # for e2 in label: y[e2] = 1.0
        y = [1] + [0] * self.p.neg_sampe_ratio
        return torch.FloatTensor(y)






class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:	The triples used for evaluating the model
    params:		Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)