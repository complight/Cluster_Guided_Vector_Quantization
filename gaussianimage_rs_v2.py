from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
from pytorch_msssim import SSIM
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import (
    UniformQuantizer, VectorQuantizer, FakeQuantizationHalf,
    compress_matrix_flatten_categorical, get_np_size
)
from optimizer import Adan


# ---------------------------------------------------------------------------
# Tile-based Multi-Codebook Quantizers
# ---------------------------------------------------------------------------

class TiledVectorQuantizer(nn.Module):
    """
    将高斯点按 2D 位置分成 tile_h × tile_w 个 tile，
    每个 tile 独立维护一个 VectorQuantizer codebook。
    
    Args:
        num_tiles_h: 竖直方向 tile 数量
        num_tiles_w: 水平方向 tile 数量
        codebook_dim: 每个向量的维度
        codebook_size: 每个 codebook 的条目数
        num_quantizers: residual VQ 层数 (1 = plain VQ)
        kmeans_iters: kmeans 初始化迭代数
    """
    def __init__(self, num_tiles_h=4, num_tiles_w=4,
                 codebook_dim=3, codebook_size=64,
                 num_quantizers=2, kmeans_iters=5):
        super().__init__()
        self.num_tiles_h = num_tiles_h
        self.num_tiles_w = num_tiles_w
        self.num_tiles = num_tiles_h * num_tiles_w
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        # 为每个 tile 创建一个独立的 VectorQuantizer
        self.tile_quantizers = nn.ModuleList([
            VectorQuantizer(
                num_quantizers=num_quantizers,
                codebook_dim=codebook_dim,
                codebook_size=codebook_size,
                kmeans_iters=kmeans_iters,
                vector_type="vector"
            )
            for _ in range(self.num_tiles)
        ])

    def _assign_tiles(self, xyz_norm):
        """
        根据归一化坐标 xyz_norm ∈ [-1, 1]^2 分配 tile id。
        
        Args:
            xyz_norm: (N, 2) tensor, 值域 [-1, 1]
        Returns:
            tile_ids: (N,) LongTensor, 每个点所属 tile 的全局索引
        """
        # 映射到 [0, num_tiles_h/w)
        row = ((xyz_norm[:, 1] + 1.0) / 2.0 * self.num_tiles_h).long().clamp(0, self.num_tiles_h - 1)
        col = ((xyz_norm[:, 0] + 1.0) / 2.0 * self.num_tiles_w).long().clamp(0, self.num_tiles_w - 1)
        return row * self.num_tiles_w + col  # (N,)

    def forward(self, features, xyz_norm):
        """
        Args:
            features : (N, C)  要量化的特征
            xyz_norm : (N, 2)  对应的归一化高斯中心坐标 (tanh 后的值)
        Returns:
            dequant  : (N, C)  反量化特征
            vq_loss  : scalar
            bits     : int (eval 阶段统计, train 时为 0)
        """
        tile_ids = self._assign_tiles(xyz_norm)  # (N,)
        dequant = torch.zeros_like(features)
        total_vq_loss = 0.0
        total_bits = 0

        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            feat_t = features[mask]           # (n_t, C)
            quant_t, loss_t, bits_t = self.tile_quantizers[t](feat_t)
            dequant[mask] = quant_t
            total_vq_loss = total_vq_loss + loss_t
            total_bits += bits_t

        return dequant, total_vq_loss, total_bits

    def compress(self, features, xyz_norm):
        """返回每个点的 dequant 特征和其 tile 内 embed_index。"""
        tile_ids = self._assign_tiles(xyz_norm)
        dequant = torch.zeros_like(features)
        # embed_index 存储每个点的量化 index（residual VQ 时为多列）
        index_list = [None] * features.shape[0]

        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            feat_t = features[mask]
            quant_t, idx_t = self.tile_quantizers[t].compress(feat_t)
            dequant[mask] = quant_t
            indices = mask.nonzero(as_tuple=False).squeeze(1)
            for local_i, global_i in enumerate(indices):
                index_list[global_i.item()] = (t, idx_t[local_i])

        return dequant, tile_ids, index_list

    def decompress(self, tile_ids, index_list, num_points, device):
        """从 tile_ids 和 index_list 重建特征。"""
        # 逐 tile 批量解码
        recon = torch.zeros(num_points, self.codebook_dim, device=device)
        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            indices = mask.nonzero(as_tuple=False).squeeze(1)
            embed_idx = torch.stack([index_list[i.item()][1] for i in indices], dim=0)
            recon[mask] = self.tile_quantizers[t].decompress(embed_idx)
        return recon

    def codebook_bits(self):
        """返回所有 tile codebook 占用的总 bit 数。"""
        bits = 0
        for tq in self.tile_quantizers:
            if tq.num_quantizers == 1:
                bits += tq.quantizer._codebook.embed.numel() * \
                        torch.finfo(tq.quantizer._codebook.embed.dtype).bits
            else:
                for layer in tq.quantizer.layers:
                    bits += layer._codebook.embed.numel() * \
                            torch.finfo(layer._codebook.embed.dtype).bits
        return bits


class TiledUniformQuantizer(nn.Module):
    """
    将高斯点按 2D 位置分成 tile_h × tile_w 个 tile，
    每个 tile 独立维护一组 (scale, beta) 的 UniformQuantizer。
    
    Args:
        num_tiles_h / num_tiles_w : tile 划分数
        signed, bits, num_channels : 同 UniformQuantizer
    """
    def __init__(self, num_tiles_h=4, num_tiles_w=4,
                 signed=False, bits=6, num_channels=2):
        super().__init__()
        self.num_tiles_h = num_tiles_h
        self.num_tiles_w = num_tiles_w
        self.num_tiles = num_tiles_h * num_tiles_w
        self.num_channels = num_channels

        self.tile_quantizers = nn.ModuleList([
            UniformQuantizer(signed=signed, bits=bits,
                             learned=True, num_channels=num_channels)
            for _ in range(self.num_tiles)
        ])

    def _assign_tiles(self, xyz_norm):
        row = ((xyz_norm[:, 1] + 1.0) / 2.0 * self.num_tiles_h).long().clamp(0, self.num_tiles_h - 1)
        col = ((xyz_norm[:, 0] + 1.0) / 2.0 * self.num_tiles_w).long().clamp(0, self.num_tiles_w - 1)
        return row * self.num_tiles_w + col

    def _init_data(self, features, xyz_norm):
        """用实际数据初始化每个 tile 的 scale/beta。"""
        tile_ids = self._assign_tiles(xyz_norm)
        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            self.tile_quantizers[t]._init_data(features[mask])

    def forward(self, features, xyz_norm):
        """
        Args:
            features : (N, C)
            xyz_norm : (N, 2)
        Returns:
            dequant, entropy_loss, bits
        """
        tile_ids = self._assign_tiles(xyz_norm)
        dequant = torch.zeros_like(features)
        total_loss = 0.0
        total_bits = 0

        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            feat_t = features[mask]
            dq_t, loss_t, bits_t = self.tile_quantizers[t](feat_t)
            dequant[mask] = dq_t
            total_loss = total_loss + loss_t
            total_bits += bits_t

        return dequant, total_loss, total_bits

    def compress(self, features, xyz_norm):
        tile_ids = self._assign_tiles(xyz_norm)
        quant_codes = torch.zeros_like(features)
        dequant = torch.zeros_like(features)

        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            code_t, dq_t = self.tile_quantizers[t].compress(features[mask])
            quant_codes[mask] = code_t
            dequant[mask] = dq_t

        return quant_codes, dequant, tile_ids

    def decompress(self, quant_codes, tile_ids):
        dequant = torch.zeros_like(quant_codes)
        for t in range(self.num_tiles):
            mask = (tile_ids == t)
            if mask.sum() == 0:
                continue
            dequant[mask] = self.tile_quantizers[t].decompress(quant_codes[mask])
        return dequant

    def codebook_bits(self):
        """返回所有 tile 的 scale+beta 所占 bit 数。"""
        bits = 0
        for tq in self.tile_quantizers:
            bits += tq.scale.numel() * torch.finfo(tq.scale.dtype).bits
            bits += tq.beta.numel() * torch.finfo(tq.beta.dtype).bits
        return bits


# ---------------------------------------------------------------------------
# GaussianImage_RS_MultiBook
# ---------------------------------------------------------------------------

class GaussianImage_RS_MultiBook(nn.Module):
    """
    基于 GaussianImage_RS 的多 codebook 版本。

    关键改动：
      - features_dc 使用 TiledVectorQuantizer（按位置分 tile，每 tile 独立 codebook）
      - scaling / rotation 使用 TiledUniformQuantizer（按位置分 tile，每 tile 独立 scale+beta）
      - 所有需要坐标的量化函数额外接受 xyz_norm 参数
    
    Args (kwargs):
        num_points, H, W, BLOCK_W, BLOCK_H, device : 同原版
        quantize     : bool
        opt_type     : "adam" | "adan"
        lr           : float
        tile_h       : 竖直 tile 数 (default 4)
        tile_w       : 水平 tile 数 (default 4)
        codebook_size: VQ codebook 大小 (default 8)
        vq_num_quantizers: residual VQ 层数 (default 2)
    """

    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        # ---------- Gaussian 参数 ----------
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.quantize = kwargs["quantize"]

        # ---------- tile 划分参数 ----------
        self.tile_h = kwargs.get("tile_h", 4)
        self.tile_w = kwargs.get("tile_w", 4)
        codebook_size = kwargs.get("codebook_size", 8)
        vq_num_quantizers = kwargs.get("vq_num_quantizers", 2)

        # ---------- Quantizers ----------
        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply

            # feature_dc: TiledVectorQuantizer
            self.features_dc_quantizer = TiledVectorQuantizer(
                num_tiles_h=self.tile_h,
                num_tiles_w=self.tile_w,
                codebook_dim=3,
                codebook_size=codebook_size,
                num_quantizers=vq_num_quantizers,
                kmeans_iters=5
            )

            # scaling: TiledUniformQuantizer (2 channels)
            self.scaling_quantizer = TiledUniformQuantizer(
                num_tiles_h=self.tile_h,
                num_tiles_w=self.tile_w,
                signed=False, bits=6, num_channels=2
            )

            # rotation: TiledUniformQuantizer (1 channel)
            self.rotation_quantizer = TiledUniformQuantizer(
                num_tiles_h=self.tile_h,
                num_tiles_w=self.tile_w,
                signed=False, bits=6, num_channels=1
            )

        # ---------- Optimizer ----------
        if kwargs.get("opt_type", "adam") == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5
        )

    # ------------------------------------------------------------------
    # 数据初始化（在 pre-quantize finetune 之前调用）
    # ------------------------------------------------------------------
    def _init_data(self):
        xyz_norm = self.get_xyz.detach()
        self.scaling_quantizer._init_data(self._scaling.detach(), xyz_norm)
        self.rotation_quantizer._init_data(self.get_rotation.detach(), xyz_norm)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def get_scaling(self):
        return torch.abs(self._scaling + self.bound)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) * 2 * math.pi

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self._opacity

    # ------------------------------------------------------------------
    # 标准前向（无量化）
    # ------------------------------------------------------------------
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            self.get_xyz, self.get_scaling, self.get_rotation,
            self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self.get_opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    # ------------------------------------------------------------------
    # 量化前向（训练阶段使用 fake-quant）
    # ------------------------------------------------------------------
    def forward_quantize(self):
        xyz_norm = self.get_xyz  # tanh 后的归一化坐标，作为 tile 分配依据

        # --- position: half-precision fake quant ---
        m_bit = 16 * self.init_num_points * 2
        means = torch.tanh(self.xyz_quantizer(self._xyz))

        # --- scaling ---
        scaling, l_vqs, s_bit = self.scaling_quantizer(self._scaling, xyz_norm.detach())
        scaling = torch.abs(scaling + self.bound)

        # --- rotation ---
        rotation, l_vqr, r_bit = self.rotation_quantizer(self.get_rotation, xyz_norm.detach())

        # --- feature_dc ---
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features, xyz_norm.detach())

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            means, scaling, rotation, self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        vq_loss = l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss,
                "unit_bit": [m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    # ------------------------------------------------------------------
    # 压缩 / 解压缩（不含熵编码）
    # ------------------------------------------------------------------
    def compress_wo_ec(self):
        xyz_norm = self.get_xyz.detach()

        quant_scaling, _, scaling_tile_ids = self.scaling_quantizer.compress(
            self._scaling.detach(), xyz_norm)
        quant_rotation, _, rotation_tile_ids = self.rotation_quantizer.compress(
            self.get_rotation.detach(), xyz_norm)
        _, feature_dc_tile_ids, feature_dc_index_list = self.features_dc_quantizer.compress(
            self.get_features.detach(), xyz_norm)

        return {
            "xyz": self._xyz.half(),
            "quant_scaling": quant_scaling,
            "scaling_tile_ids": scaling_tile_ids,
            "quant_rotation": quant_rotation,
            "rotation_tile_ids": rotation_tile_ids,
            "feature_dc_tile_ids": feature_dc_tile_ids,
            "feature_dc_index_list": feature_dc_index_list,
        }

    def decompress_wo_ec(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        means = torch.tanh(xyz.float())

        scaling = self.scaling_quantizer.decompress(
            encoding_dict["quant_scaling"], encoding_dict["scaling_tile_ids"])
        scaling = torch.abs(scaling + self.bound)

        rotation = self.rotation_quantizer.decompress(
            encoding_dict["quant_rotation"], encoding_dict["rotation_tile_ids"])

        colors = self.features_dc_quantizer.decompress(
            encoding_dict["feature_dc_tile_ids"],
            encoding_dict["feature_dc_index_list"],
            num_points=self.init_num_points,
            device=self.device
        )

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def analysis_wo_ec(self, encoding_dict):
        """无熵编码情况下的 bit 统计（与原版 analysis_wo_ec 对应）。"""
        quant_scaling = encoding_dict["quant_scaling"]
        quant_rotation = encoding_dict["quant_rotation"]
        feature_dc_index_list = encoding_dict["feature_dc_index_list"]

        # --- codebook bits ---
        scaling_codebook_bits = self.scaling_quantizer.codebook_bits()
        rotation_codebook_bits = self.rotation_quantizer.codebook_bits()
        feature_dc_codebook_bits = self.features_dc_quantizer.codebook_bits()
        initial_bits = scaling_codebook_bits + rotation_codebook_bits + feature_dc_codebook_bits

        # --- index bits (raw, no entropy coding) ---
        qs_np = quant_scaling.cpu().numpy()
        qr_np = quant_rotation.cpu().numpy()

        # feature_dc: 取 embed index 的最大值估算 bit 宽
        all_dc_indices = []
        for item in feature_dc_index_list:
            if item is not None:
                idx = item[1]
                all_dc_indices.append(idx.cpu().numpy().flatten())
        if all_dc_indices:
            all_dc_np = np.concatenate(all_dc_indices)
            dc_index_max = np.max(all_dc_np) if all_dc_np.size > 0 else 1
            dc_max_bit = max(1, int(np.ceil(np.log2(dc_index_max + 1))))
            dc_index_total = all_dc_np.size
        else:
            dc_max_bit, dc_index_total = 1, 0

        position_bits = self._xyz.numel() * 16
        scaling_bits = scaling_codebook_bits + qs_np.size * 6
        rotation_bits = rotation_codebook_bits + qr_np.size * 6
        feature_dc_bits = feature_dc_codebook_bits + dc_index_total * dc_max_bit

        total_bits = initial_bits + position_bits + qs_np.size * 6 + qr_np.size * 6 + dc_index_total * dc_max_bit

        bpp = total_bits / self.H / self.W
        return {
            "bpp": bpp,
            "position_bpp": position_bits / self.H / self.W,
            "scaling_bpp": scaling_bits / self.H / self.W,
            "rotation_bpp": rotation_bits / self.H / self.W,
            "cholesky_bpp": (scaling_bits + rotation_bits) / self.H / self.W,
            "feature_dc_bpp": feature_dc_bits / self.H / self.W,
        }

    # ------------------------------------------------------------------
    # 压缩 / 解压缩（含熵编码，对应原版 compress / decompress / analysis）
    # ------------------------------------------------------------------
    def compress(self):
        xyz_norm = self.get_xyz.detach()

        _, scaling_dq, scaling_tile_ids = self.scaling_quantizer.compress(
            self._scaling.detach(), xyz_norm)
        _, rotation_dq, rotation_tile_ids = self.rotation_quantizer.compress(
            self.get_rotation.detach(), xyz_norm)
        _, feature_dc_tile_ids, feature_dc_index_list = self.features_dc_quantizer.compress(
            self.get_features.detach(), xyz_norm)

        # 对 scaling/rotation 的 dequant 值做熵编码（与原版一致：先 round 成整数索引）
        scaling_index = self.scaling_quantizer.compress(
            self._scaling.detach(), xyz_norm)[0]  # quant codes (integers)
        rotation_index = self.rotation_quantizer.compress(
            self.get_rotation.detach(), xyz_norm)[0]

        return {
            "xyz": self._xyz.half(),
            "scaling_index": scaling_index,
            "scaling_tile_ids": scaling_tile_ids,
            "rotation_index": rotation_index,
            "rotation_tile_ids": rotation_tile_ids,
            "feature_dc_tile_ids": feature_dc_tile_ids,
            "feature_dc_index_list": feature_dc_index_list,
        }

    def decompress(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        means = torch.tanh(xyz.float())

        scaling = self.scaling_quantizer.decompress(
            encoding_dict["scaling_index"], encoding_dict["scaling_tile_ids"])
        scaling = torch.abs(scaling + self.bound)

        rotation = self.rotation_quantizer.decompress(
            encoding_dict["rotation_index"], encoding_dict["rotation_tile_ids"])

        colors = self.features_dc_quantizer.decompress(
            encoding_dict["feature_dc_tile_ids"],
            encoding_dict["feature_dc_index_list"],
            num_points=self.init_num_points,
            device=self.device
        )

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def analysis(self, encoding_dict):
        """含熵编码的 bit 统计（对应原版 analysis）。"""
        scaling_index = encoding_dict["scaling_index"]
        rotation_index = encoding_dict["rotation_index"]
        feature_dc_index_list = encoding_dict["feature_dc_index_list"]

        # 熵编码：每个 tile 分别压缩，然后合计 bit 数
        scaling_bits_ec, rotation_bits_ec, feature_dc_bits_ec = 0, 0, 0
        scaling_header_bits, rotation_header_bits, feature_dc_header_bits = 0, 0, 0

        tile_ids_s = self.scaling_quantizer._assign_tiles(self.get_xyz.detach())
        tile_ids_r = self.rotation_quantizer._assign_tiles(self.get_xyz.detach())

        for t in range(self.num_tiles if hasattr(self, 'num_tiles') else self.tile_h * self.tile_w):
            # scaling
            mask_s = (tile_ids_s == t)
            if mask_s.sum() > 0:
                codes_s = scaling_index[mask_s].int().flatten().tolist()
                comp_s, hist_s, uniq_s = compress_matrix_flatten_categorical(codes_s)
                scaling_bits_ec += get_np_size(comp_s) * 8
                scaling_header_bits += (get_np_size(hist_s) + get_np_size(uniq_s)) * 8

            # rotation
            mask_r = (tile_ids_r == t)
            if mask_r.sum() > 0:
                codes_r = rotation_index[mask_r].int().flatten().tolist()
                comp_r, hist_r, uniq_r = compress_matrix_flatten_categorical(codes_r)
                rotation_bits_ec += get_np_size(comp_r) * 8
                rotation_header_bits += (get_np_size(hist_r) + get_np_size(uniq_r)) * 8

        # feature_dc：逐 tile 压缩
        tile_ids_dc = self.features_dc_quantizer._assign_tiles(self.get_xyz.detach())
        for t in range(self.tile_h * self.tile_w):
            tile_indices = []
            for item in feature_dc_index_list:
                if item is not None and item[0] == t:
                    tile_indices.append(item[1].cpu().numpy().flatten())
            if tile_indices:
                all_idx = np.concatenate(tile_indices)
                comp_dc, hist_dc, uniq_dc = compress_matrix_flatten_categorical(all_idx.tolist())
                feature_dc_bits_ec += get_np_size(comp_dc) * 8
                feature_dc_header_bits += (get_np_size(hist_dc) + get_np_size(uniq_dc)) * 8

        scaling_codebook_bits = self.scaling_quantizer.codebook_bits()
        rotation_codebook_bits = self.rotation_quantizer.codebook_bits()
        feature_dc_codebook_bits = self.features_dc_quantizer.codebook_bits()

        position_bits = self._xyz.numel() * 16
        scaling_bits = scaling_codebook_bits + scaling_header_bits + scaling_bits_ec
        rotation_bits = rotation_codebook_bits + rotation_header_bits + rotation_bits_ec
        feature_dc_bits = feature_dc_codebook_bits + feature_dc_header_bits + feature_dc_bits_ec

        total_bits = position_bits + scaling_bits + rotation_bits + feature_dc_bits

        bpp = total_bits / self.H / self.W
        return {
            "bpp": bpp,
            "position_bpp": position_bits / self.H / self.W,
            "scaling_bpp": scaling_bits / self.H / self.W,
            "rotation_bpp": rotation_bits / self.H / self.W,
            "cholesky_bpp": (scaling_bits + rotation_bits) / self.H / self.W,
            "feature_dc_bpp": feature_dc_bits / self.H / self.W,
        }
