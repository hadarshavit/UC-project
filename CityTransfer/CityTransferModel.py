# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 0009 8:49
# @Author  : Binjie Zhang (bj_zhang@seu.edu.cn)
# @File    : _discarded_CityTransfer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from torch.nn.modules import pooling
from utility.log_helper import logging


# Define the original Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self._define_autoencoder(in_dim, out_dim)

    def _define_autoencoder(self, in_dim, out_dim):
        self.mid_dim = math.ceil(math.sqrt(in_dim * out_dim))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.mid_dim),
            nn.SELU(),
            nn.Linear(self.mid_dim, out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim),
            nn.SELU(),
            nn.Linear(self.mid_dim, in_dim),
        )

    def forward(self, x):
        # Equation (15 & 16)
        encoded_x = self.encoder(x)
        # Equation (17 & 18)
        decoded_x = self.decoder(encoded_x)
        return encoded_x, decoded_x

# Define the  ThreeLayer Denoising Model
class TwoLayersAutoencoder(AutoEncoder):
    def _define_autoencoder(self, in_dim, out_dim):
        self.mid_dim1 = math.ceil(math.sqrt(in_dim * out_dim))
        print(self.mid_dim1)
        self.mid_dim2 = math.ceil(math.sqrt(self.mid_dim1 * out_dim))
        print(self.mid_dim2)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, in_dim),
        )

# Define SparseAutoencoder
class FourLayerAutoencoder(AutoEncoder):
    def _define_autoencoder(self, in_dim, out_dim):
        self.mid_dim1 = math.ceil(math.sqrt(in_dim * out_dim))
        self.mid_dim2 = math.ceil(math.sqrt(self.mid_dim1 * out_dim))
        self.mid_dim3 = math.ceil(math.sqrt(self.mid_dim2 * out_dim))
        print('DEFINE FOUR', self.mid_dim1, self.mid_dim2, self.mid_dim3)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SELU(),
            nn.Linear(in_dim, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim3),
            nn.SELU(),
            nn.Linear(self.mid_dim3, out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim3),
            nn.SELU(),
            nn.Linear(self.mid_dim3, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, in_dim),
            nn.SELU(),
            nn.Linear(in_dim, in_dim)
        )

# Define SparseAutoencoder
class SixLayesAutoencoder(AutoEncoder):
    def _define_autoencoder(self, in_dim, out_dim):
        self.mid_dim1 = math.ceil(math.sqrt(in_dim * out_dim))
        self.mid_dim2 = math.ceil(math.sqrt(self.mid_dim1 * out_dim))
        self.mid_dim3 = math.ceil(math.sqrt(self.mid_dim2 * out_dim))
        self.mid_dim4 = math.ceil(math.sqrt(self.mid_dim3 * out_dim))
        self.mid_dim5 = math.ceil(math.sqrt(self.mid_dim4 * out_dim))
        print('DEFINE SIX', self.mid_dim1, self.mid_dim2, self.mid_dim3,self.mid_dim4,self.mid_dim5)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SELU(),
            nn.Linear(in_dim, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim3),
            nn.SELU(),
            nn.Linear(self.mid_dim3, self.mid_dim4),
            nn.SELU(),
            nn.Linear(self.mid_dim4, self.mid_dim5),
            nn.SELU(),
            nn.Linear(self.mid_dim5, out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim5),
            nn.SELU(),
            nn.Linear(self.mid_dim5, self.mid_dim4),
            nn.SELU(),
            nn.Linear(self.mid_dim4, self.mid_dim3),
            nn.SELU(),
            nn.Linear(self.mid_dim3, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, in_dim),
            nn.SELU(),
            nn.Linear(in_dim, in_dim)
        )

# Define SparseAutoencoder
class EightLayesAutoencoder(AutoEncoder):
    def _define_autoencoder(self, in_dim, out_dim):
        self.mid_dim1 = math.ceil(math.sqrt(in_dim * out_dim))
        self.mid_dim2 = math.ceil(math.sqrt(self.mid_dim1 * out_dim))
        self.mid_dim3 = math.ceil(math.sqrt(self.mid_dim2 * out_dim))
        self.mid_dim4 = math.ceil(math.sqrt(self.mid_dim3 * out_dim))
        self.mid_dim5 = math.ceil(math.sqrt(self.mid_dim4 * out_dim))
        self.mid_dim6 = math.ceil(math.sqrt(self.mid_dim5 * out_dim))
        self.mid_dim7 = math.ceil(math.sqrt(self.mid_dim6 * out_dim))
        print('DEFINE eight', self.mid_dim1, self.mid_dim2, self.mid_dim3,self.mid_dim4,self.mid_dim5,self.mid_dim6,self.mid_dim7)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SELU(),
            nn.Linear(in_dim, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim3),
            nn.SELU(),
            nn.Linear(self.mid_dim3, self.mid_dim4),
            nn.SELU(),
            nn.Linear(self.mid_dim4, self.mid_dim5),
            nn.SELU(),
            nn.Linear(self.mid_dim5, self.mid_dim6),
            nn.SELU(),
            nn.Linear(self.mid_dim6, self.mid_dim7),
            nn.SELU(),
            nn.Linear(self.mid_dim7, out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, self.mid_dim7),
            nn.SELU(),
            nn.Linear(self.mid_dim7, self.mid_dim6),
            nn.SELU(),
            nn.Linear(self.mid_dim6, self.mid_dim5),
            nn.SELU(),
            nn.Linear(self.mid_dim5, self.mid_dim4),
            nn.SELU(),
            nn.Linear(self.mid_dim4, self.mid_dim3),
            nn.SELU(),
            nn.Linear(self.mid_dim3, self.mid_dim2),
            nn.SELU(),
            nn.Linear(self.mid_dim2, self.mid_dim1),
            nn.SELU(),
            nn.Linear(self.mid_dim1, in_dim),
            nn.SELU(),
            nn.Linear(in_dim, in_dim)
        )


class CityTransfer(nn.Module):
    def __init__(self, args, feature_dim, n_source_grid, n_target_grid):
        super(CityTransfer, self).__init__()
        self.args = args

        # auto encoder
        self.auto_encoder = nn.ModuleList()
        if args.autoencoder == 'TwoLayersAC':
            self.auto_encoder.append(TwoLayersAutoencoder(feature_dim, self.args.auto_encoder_dim))  # source
            self.auto_encoder.append(TwoLayersAutoencoder(feature_dim, self.args.auto_encoder_dim))  # target
        elif args.autoencoder == 'FourLayersAC':
            self.auto_encoder.append(FourLayerAutoencoder(feature_dim, self.args.auto_encoder_dim))  # source
            self.auto_encoder.append(FourLayerAutoencoder(feature_dim, self.args.auto_encoder_dim))  # target
        elif args.autoencoder == 'Default':
            self.auto_encoder.append(AutoEncoder(feature_dim, self.args.auto_encoder_dim))  # source
            self.auto_encoder.append(AutoEncoder(feature_dim, self.args.auto_encoder_dim))  # target
        elif args.autoencoder == 'SixLayersAC':
            self.auto_encoder.append(SixLayesAutoencoder(feature_dim, self.args.auto_encoder_dim))  # source
            self.auto_encoder.append(SixLayesAutoencoder(feature_dim, self.args.auto_encoder_dim))  # target
        elif args.autoencoder == 'EightLayersAC':
            self.auto_encoder.append(EightLayesAutoencoder(feature_dim, self.args.auto_encoder_dim))  # source
            self.auto_encoder.append(EightLayesAutoencoder(feature_dim, self.args.auto_encoder_dim))  # target
        else:
            assert False

        # matrix factorization
        self.u = nn.Parameter(torch.Tensor(len(self.args.enterprise), self.args.auto_encoder_dim))
        self.b = nn.Parameter(torch.Tensor(len(self.args.enterprise), 1))
        self.e_source = nn.Parameter(torch.Tensor(n_source_grid, 1))
        self.e_target = nn.Parameter(torch.Tensor(n_target_grid, 1))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.b)
        nn.init.xavier_uniform_(self.e_source)
        nn.init.xavier_uniform_(self.e_target)

    def encode(self, origin_feature, ae_type):
        if ae_type == 's':
            encoded_x, decoded_x = self.auto_encoder[0](origin_feature)  # source
        elif ae_type == 't':
            encoded_x, decoded_x = self.auto_encoder[1](origin_feature)  # target
        else:
            logging.error('未定义类型')
            exit(1)
        return encoded_x, decoded_x

    def cal_auto_encoder_loss(self, grid_feature, ae_type):
        _, decoded_grid = self.encode(grid_feature, ae_type)
        # Equation (19)
        loss_ae = F.mse_loss(grid_feature, decoded_grid, reduction='sum')
        # loss_ae = F.mse_loss(grid_feature, decoded_grid, reduction='mean')
        return loss_ae

    def cal_inter_city_loss(self, score, source_feature, target_feature):
        encoded_source, _ = self.encode(source_feature, 's')
        encoded_target, _ = self.encode(target_feature, 't')
        # Equation (14)
        loss = torch.sum(score * torch.sum(torch.pow(encoded_source - encoded_target, 2), dim=2))
        # loss = torch.mean(score * torch.mean(torch.pow(encoded_source - encoded_target, 2), dim=2))
        return loss

    def cal_prediction_score(self, enterprise_index, grid_index, grid_feature, grid_type):
        encoded_feature, _ = self.encode(grid_feature, grid_type)
        enterprise_feature = self.u[enterprise_index]
        enterprise_bias = self.b[enterprise_index]

        if grid_type == 's':
            s = self.e_source[grid_index]
            g = len(grid_index)
            # print(grid_index)
            grid_bias = self.e_source[grid_index].reshape(-1, len(grid_index))
        elif grid_type == 't':
            grid_bias = self.e_target[grid_index].reshape(-1, len(grid_index))
        else:
            logging.error('未定义类型')
            exit(1)
        # Equation (10 & 11)
        score = torch.matmul(enterprise_feature.unsqueeze(1), encoded_feature.permute(0, 2, 1)).squeeze(1) + \
                enterprise_bias + grid_bias
        return score

    def cal_prediction_loss(self, enterprise_index, grid_index, grid_feature, grid_type, real_score):
        score = self.cal_prediction_score(enterprise_index, grid_index, grid_feature, grid_type)
        # origin_index = torch.argsort(real_score, descending=True)
        # predict_index = torch.argsort(score, descending=True)
        # Equation (12)
        loss = F.mse_loss(score, real_score, reduction='sum')
        # loss = F.mse_loss(score, real_score, reduction='mean')
        return loss

    def prediction(self, target_enterprise_index, grid_index, grid_feature):
        encoded_feature, _ = self.encode(grid_feature, 't')
        enterprise_feature = self.u[target_enterprise_index]
        enterprise_bias = self.b[target_enterprise_index]
        grid_bias = self.e_target[grid_index].reshape(-1, len(grid_index))

        score = (torch.matmul(enterprise_feature, encoded_feature.T) + enterprise_bias + grid_bias).squeeze()
        return score

    def forward(self, mode, *inputs):
        if mode == 'cal_auto_encoder_loss':
            return self.cal_auto_encoder_loss(*inputs)
        elif mode == 'cal_inter_city_loss':
            return self.cal_inter_city_loss(*inputs)
        elif mode == 'cal_prediction_loss':
            return self.cal_prediction_loss(*inputs)
        elif mode == 'prediction':
            return self.prediction(*inputs)
        else:
            logging.error('wrong parameters!')
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="City Transfer Args.")
    args = parser.parse_args()
    args.auto_encoder_dim = 9
    args.enterprise = ['a', 'b', 'c']
    c = CityTransfer(args, 5, 1000, 1000)
    aa = torch.Tensor([[0.15, 0.71, 0.5, 0.4, 0.3], [0.15, 0.71, 0.5, 0.4, 0.3], [0.15, 0.71, 0.5, 0.4, 0.3]])
    bb = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1]])
    ab = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1], [0.75, 0.61, 0.4, 0.9, 0.1]])
    cc = torch.Tensor([[0.75, 0.61, 0.4, 0.9, 0.1, 0.61, 0.4, 0.9, 0.1], [0.2, 0.7, 0.4, 0.4, 0.1, 0.9, 0.4, 0.2, 0.1]])
    dd = torch.Tensor([[0.8, 0.7], [0.3, 0.1], [0.6, 0.9]])
    # res = c.cal_prediction_loss([1, 2, 0], [30, 40], ab, 's', dd)
    res2 = c.cal_prediction_score([1, 2, 0], [30, 40], ab, 't')
    print(res2)
