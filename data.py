import numpy as np
import torch
from torch.utils.data import Dataset


class TSDataset(Dataset):
    def __init__(self, data, sql_len=1):
        self.sql_len = sql_len
        if self.sql_len > 1:
            data = self.transform(data)
        self.data = data

    def transform(self, data):
        """
        构造时序数据
        """
        output = []
        for i in range(len(data) - self.sql_len + 1):
            output.append(data[i:i+self.sql_len])
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        if self.sql_len == 1:
            data = data.unsqueeze(0)
        return data


def enrich_raw_ts(df, b, f, model, norm=True):
    # Step 1
    step_length_1 = b // 2
    new_ts_1_nor = []
    new_ts_1_don = []
    i = 0
    while i <= df.shape[0] - b:
        # 计算窗口向量的二范数
        norms = []
        for j in range(df.shape[1]):
            array = df.iloc[i:i+b, j].to_numpy().reshape(-1, 1)
            euclidean_length = np.linalg.norm(array)
            norms.append(euclidean_length)
        new_ts_1_nor.append(norms)
        # 计算前后两窗口向量的二范数的差值
        if i == 0:
            difference_of_norms = [0] * len(norms)
        else:
            difference_of_norms = np.array(norms) - np.array(new_ts_1_nor[-2])
            difference_of_norms = difference_of_norms.tolist()
        new_ts_1_don.append(difference_of_norms)
        i += step_length_1
    # 归一化
    if norm:
        epsilon = 1e-7
        new_ts_1_nor = np.array(new_ts_1_nor)
        nor_min_col = new_ts_1_nor.min(axis=0)
        nor_max_col = new_ts_1_nor.max(axis=0)
        new_ts_1_nor = (new_ts_1_nor - nor_min_col) / (nor_max_col - nor_min_col + epsilon)
        new_ts_1_nor = new_ts_1_nor.tolist()
        new_ts_1_don = np.array(new_ts_1_don)
        don_min_col = new_ts_1_don.min(axis=0)
        don_max_col = new_ts_1_don.max(axis=0)
        new_ts_1_don = (new_ts_1_don - don_min_col) / (don_max_col - don_min_col + epsilon)
        new_ts_1_don = new_ts_1_don.tolist()
    new_ts_1 = []
    for i in range(len(new_ts_1_nor)):
        new_ts_1.append([new_ts_1_nor[i], new_ts_1_don[i]])

    # Step 2
    step_length_2 = f // 2
    new_ts_2 = []
    i = 0
    while i <= len(new_ts_1) - f:
        array = new_ts_1[i:i+f]
        norms = []
        difference_of_norms = []
        for s in range(f):
            norms.append(array[s][0])
            difference_of_norms.append(array[s][1])
        # 计算8个统计特征
        statistical_features = [[] for _ in range(16)]
        for j in range(df.shape[1]):
            norm = np.array([row[j] for row in norms])
            norm_mean = np.mean(norm)
            statistical_features[0].append(norm_mean)
            norm_min = np.min(norm)
            statistical_features[1].append(norm_min)
            norm_max = np.max(norm)
            statistical_features[2].append(norm_max)
            norm_q25 = np.percentile(norm, 25)
            statistical_features[3].append(norm_q25)
            norm_q50 = np.percentile(norm, 50)
            statistical_features[4].append(norm_q50)
            norm_q75 = np.percentile(norm, 75)
            statistical_features[5].append(norm_q75)
            norm_std = np.std(norm)
            statistical_features[6].append(norm_std)
            norm_p2p = norm_max - norm_min
            statistical_features[7].append(norm_p2p)
            difference_of_norm = np.array([row[j] for row in difference_of_norms])
            difference_of_norm_mean = np.mean(difference_of_norm)
            statistical_features[8].append(difference_of_norm_mean)
            difference_of_norm_min = np.min(difference_of_norm)
            statistical_features[9].append(difference_of_norm_min)
            difference_of_norm_max = np.max(difference_of_norm)
            statistical_features[10].append(difference_of_norm_max)
            difference_of_norm_q25 = np.percentile(difference_of_norm, 25)
            statistical_features[11].append(difference_of_norm_q25)
            difference_of_norm_q50 = np.percentile(difference_of_norm, 50)
            statistical_features[12].append(difference_of_norm_q50)
            difference_of_norm_q75 = np.percentile(difference_of_norm, 75)
            statistical_features[13].append(difference_of_norm_q75)
            difference_of_norm_std = np.std(difference_of_norm)
            statistical_features[14].append(difference_of_norm_std)
            difference_of_norm_p2p = difference_of_norm_max - difference_of_norm_min
            statistical_features[15].append(difference_of_norm_p2p)
        new_ts_2.append(statistical_features)
        i += step_length_2

    # 向量拼接
    if model == 'lstm':
        new_ts_2_con = []
        for two_d_list in new_ts_2:
            new_list = []
            for one_d_list in two_d_list:
                new_list.extend(one_d_list)
            new_ts_2_con.append(new_list)
        new_ts_2 = new_ts_2_con

    return new_ts_2
