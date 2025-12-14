import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils as pn2
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .build import MODELS
from timm.models.layers import DropPath, trunc_normal_
from .dgcnn_group import DGCNN_Grouper
from knn_cuda import KNN
from models.model_utils import edge_preserve_sampling, get_graph_feature, symmetric_sample, three_nn_upsampling  # 必须保留

def fps(pc, num):
    fps_idx = pn2.furthest_point_sample(pc, num)
    sub_pc = pn2.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

knn = KNN(k=8, transpose_mode=False)


def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        _, idx = knn(coor_k, coor_q)  # bs k np
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)

    return idx


def get_graph_feature_1(x, knn_index, x_q=None):
    k = 8
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x, x), dim=-1)
    return feature

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Offset-Attention
        x = x - attn_output

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim * 2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index=None):
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        if knn_index is not None:
            knn_f = get_graph_feature_1(norm_x, knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            x_1 = torch.cat([x_1, knn_f], dim=-1)
            x_1 = self.merge_map(x_1)

        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Stack_conv(nn.Module):
    def __init__(self, input_size, output_size, act=None):
        super(Stack_conv, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv', nn.Conv2d(input_size, output_size, 1))

        if act is not None:
            self.model.add_module('act', act)

    def forward(self, x):
        y = self.model(x)
        y = torch.cat((x, y), 1)
        return y


class Dense_conv(nn.Module):
    def __init__(self, input_size, growth_rate=64, dense_n=3, k=16):
        super(Dense_conv, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.comp = growth_rate * 2
        self.input_size = input_size

        self.first_conv = nn.Conv2d(self.input_size * 2, growth_rate, 1)

        self.input_size += self.growth_rate

        self.model = nn.Sequential()
        for i in range(dense_n - 1):
            if i == dense_n - 2:
                self.model.add_module('stack_conv_%d' % (i + 1), Stack_conv(self.input_size, self.growth_rate, None))
            else:
                self.model.add_module('stack_conv_%d' % (i + 1),
                                      Stack_conv(self.input_size, self.growth_rate, nn.ReLU()))
                self.input_size += growth_rate

    def forward(self, x):
        y = get_graph_feature(x, k=self.k)
        y = F.relu(self.first_conv(y))
        y = torch.cat((y, x.unsqueeze(3).repeat(1, 1, 1, self.k)), 1)

        y = self.model(y)
        y, _ = torch.max(y, 3)

        return y


class EF_encoder(nn.Module):
    def __init__(self, growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64], input_size=3, output_size=256):
        super(EF_encoder, self).__init__()
        self.growth_rate = growth_rate
        self.comp = growth_rate * 2
        self.dense_n = dense_n
        self.k = k
        self.hierarchy = hierarchy

        self.init_channel = 24

        self.conv1 = nn.Conv1d(input_size, self.init_channel, 1)
        self.dense_conv1 = Dense_conv(self.init_channel, self.growth_rate, self.dense_n, self.k)

        out_channel_size_1 = (self.init_channel * 2 + self.growth_rate * self.dense_n)  # 24*2 + 24*3 = 120
        self.conv2 = nn.Conv1d(out_channel_size_1 * 2, self.comp, 1)
        self.dense_conv2 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_2 = (
                    out_channel_size_1 * 2 + self.comp + self.growth_rate * self.dense_n)  # 120*2 + 48 + 24*3 = 360
        self.conv3 = nn.Conv1d(out_channel_size_2 * 2, self.comp, 1)
        self.dense_conv3 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_3 = (
                    out_channel_size_2 * 2 + self.comp + self.growth_rate * self.dense_n)  # 360*2 + 48 + 24*3 = 840
        self.conv4 = nn.Conv1d(out_channel_size_3 * 2, self.comp, 1)
        self.dense_conv4 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_4 = out_channel_size_3 * 2 + self.comp + self.growth_rate * self.dense_n  # 840*2 + 48 + 24*3 = 1800
        self.gf_conv = nn.Conv1d(out_channel_size_4, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        out_channel_size = out_channel_size_4 + 1024
        self.conv5 = nn.Conv1d(out_channel_size, 1024, 1)

        out_channel_size = out_channel_size_3 + 1024
        self.conv6 = nn.Conv1d(out_channel_size, 768, 1)

        out_channel_size = out_channel_size_2 + 768
        self.conv7 = nn.Conv1d(out_channel_size, 512, 1)

        out_channel_size = out_channel_size_1 + 512
        self.conv8 = nn.Conv1d(out_channel_size, output_size, 1)

    def forward(self, x):
        point_cloud1 = x[:, 0:3, :]
        point_cloud1 = point_cloud1.transpose(1, 2).contiguous()

        x0 = F.relu(self.conv1(x))  # 24
        x1 = F.relu(self.dense_conv1(x0))  # 24 + 24 * 3 = 96
        x1 = torch.cat((x1, x0), 1)  # 120
        x1d, _, _, point_cloud2 = edge_preserve_sampling(x1, point_cloud1, self.hierarchy[0], self.k)  # 240

        x2 = F.relu(self.conv2(x1d))  # 48
        x2 = F.relu(self.dense_conv2(x2))  # 48 + 24 * 3 = 120
        x2 = torch.cat((x2, x1d), 1)  # 120 + 240 = 360
        x2d, _, _, point_cloud3 = edge_preserve_sampling(x2, point_cloud2, self.hierarchy[1], self.k)  # 720

        x3 = F.relu(self.conv3(x2d))
        x3 = F.relu(self.dense_conv3(x3))
        x3 = torch.cat((x3, x2d), 1)
        x3d, _, _, point_cloud4 = edge_preserve_sampling(x3, point_cloud3, self.hierarchy[2], self.k)

        x4 = F.relu(self.conv4(x3d))
        x4 = F.relu(self.dense_conv4(x4))
        x4 = torch.cat((x4, x3d), 1)

        global_feat = self.gf_conv(x4)
        global_feat, _ = torch.max(global_feat, -1)
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat)).unsqueeze(2).repeat(1, 1, self.hierarchy[2])

        x4 = torch.cat((global_feat, x4), 1)
        x4 = F.relu(self.conv5(x4))
        idx, weight = three_nn_upsampling(point_cloud3, point_cloud4)
        x4 = pn2.three_interpolate(x4, idx, weight)

        x3 = torch.cat((x3, x4), 1)
        x3 = F.relu(self.conv6(x3))
        idx, weight = three_nn_upsampling(point_cloud2, point_cloud3)
        x3 = pn2.three_interpolate(x3, idx, weight)

        x2 = torch.cat((x2, x3), 1)
        x2 = F.relu(self.conv7(x2))
        idx, weight = three_nn_upsampling(point_cloud1, point_cloud2)
        x2 = pn2.three_interpolate(x2, idx, weight)

        x1 = torch.cat((x1, x2), 1)
        x1 = self.conv8(x1)
        return x1


class EF_expansion(nn.Module):
    def __init__(self, input_size, output_size=64, step_ratio=2, k=4):
        super(EF_expansion, self).__init__()
        self.step_ratio = step_ratio
        self.k = k
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_size * 2, output_size, 1)
        self.conv2 = nn.Conv2d(input_size * 2 + output_size, output_size * step_ratio, 1)
        self.conv3 = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        input_edge_feature = get_graph_feature(x, self.k, minus_center=False).permute(0, 1, 3,
                                                                                      2).contiguous()  # B C K N
        edge_feature = self.conv1(input_edge_feature)
        edge_feature = F.relu(torch.cat((edge_feature, input_edge_feature), 1))

        edge_feature = F.relu(self.conv2(edge_feature))  # B C K N
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous().view(batch_size, self.k,
                                                                          num_points * self.step_ratio,
                                                                          self.output_size).permute(0, 3, 1,
                                                                                                    2)  # B C K N

        edge_feature = self.conv3(edge_feature)
        edge_feature, _ = torch.max(edge_feature, 2)

        return edge_feature


class ECG_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, num_input, downsample_im=False, mirror_im=False, points_label=False):
        super(ECG_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        if not downsample_im:
            self.scale = int(np.ceil(num_fine / (num_coarse + num_input)))
        else:
            self.scale = int(np.ceil(num_fine / 2048))

        self.downsample_im = downsample_im
        self.mirror_im = mirror_im
        self.points_label = points_label

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.dense_feature_size = 256
        self.expand_feature_size = 64

        if points_label:
            self.input_size = 4
        else:
            self.input_size = 3

        self.encoder = EF_encoder(growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64],
                                  input_size=self.input_size, output_size=self.dense_feature_size)

        if self.scale >= 2:
            self.expansion = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
                                          step_ratio=self.scale, k=4)
            self.conv1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion = None
            self.conv1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv2 = nn.Conv1d(self.expand_feature_size, 3, 1)

    def forward(self, global_feat, point_input):
        batch_size = global_feat.size()[0]
        coarse = F.relu(self.fc1(global_feat))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(batch_size, 3, self.num_coarse)

        if self.downsample_im:
            if self.mirror_im:
                org_points_input = symmetric_sample(point_input.transpose(1, 2).contiguous(),
                                                    int((2048 - self.num_coarse) / 2))
                org_points_input = org_points_input.transpose(1, 2).contiguous()
            else:
                org_points_input = pn2.gather_operation(point_input,
                                                        pn2.furthest_point_sample(
                                                            point_input.transpose(1, 2).contiguous(),
                                                            int(2048 - self.num_coarse)))
        else:
            org_points_input = point_input

        if self.points_label:
            id0 = torch.zeros(coarse.shape[0], 1, coarse.shape[2]).cuda().contiguous()
            coarse_input = torch.cat((coarse, id0), 1)
            id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).cuda().contiguous()
            org_points_input = torch.cat((org_points_input, id1), 1)
            points = torch.cat((coarse_input, org_points_input), 2)
        else:
            points = torch.cat((coarse, org_points_input), 2)

        dense_feat = self.encoder(points)

        if self.scale >= 2:
            dense_feat = self.expansion(dense_feat)

        point_feat = F.relu(self.conv1(dense_feat))
        fine = self.conv2(point_feat)

        num_out = fine.size()[2]
        if num_out > self.num_fine:
            fine = pn2.gather_operation(fine,
                                        pn2.furthest_point_sample(fine.transpose(1, 2).contiguous(), self.num_fine))

        return coarse, fine
    
class PCT(nn.Module):

    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 num_query=224, knn_layer=-1):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.knn_layer = knn_layer

        self.grouper = DGCNN_Grouper()

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])

        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query = num_query
        
        self.decoder = ECG_decoder(num_coarse=num_query, num_fine=6144, num_input=2048, downsample_im=False, mirror_im=False, points_label=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)




    def forward(self, inpc):

        basi = inpc.size(0)
        xy, f = self.grouper(inpc.transpose(1, 2).contiguous())
        knn_index = get_knn_index(xy)
        pos = self.pos_embed(xy).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2)

        # encoder
        for i, enc in enumerate(self.encoder):
            if i < self.knn_layer:
                x = enc(x + pos, knn_index)  # B N C
            else:
                x = enc(x + pos)

        global_feature = self.increase_dim(x.transpose(1, 2))  # B 1024 N
        global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024
        coarse, fine = self.decoder(global_feature, inpc)
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()

        return coarse, fine



@MODELS.register_module()
class SGrasp(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_query = config.num_query

        self.base_model = PCT(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., num_query = self.num_query, knn_layer = self.knn_layer)

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)

        return loss_coarse, loss_fine

    def forward(self, points):
        # point cloud completion
        coarse_point_cloud, build_points = self.base_model(points)
        # cat
        inp_sparse = fps(points, self.num_query)
        sparse_pcd = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        build_points = torch.cat([build_points, points],dim=1).contiguous()
        output = (sparse_pcd, build_points)
        return output
