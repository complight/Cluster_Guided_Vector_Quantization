from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import KMeans
from quantize import (
    UniformQuantizer, VectorQuantizer, FakeQuantizationHalf,
    compress_matrix_flatten_categorical, get_np_size
)
from optimizer import Adan


def build_cluster_features(feature_dc, scaling, rotation):
    parts = [feature_dc, scaling, rotation]
    normalized = []
    for p in parts:
        mu  = p.mean(dim=0, keepdim=True)
        std = p.std(dim=0,  keepdim=True).clamp(min=1e-6)
        normalized.append((p - mu) / std)
    return torch.cat(normalized, dim=1)   # (N, 6)


def adaptive_num_clusters(num_points, min_clusters=4, max_clusters=64, scale=0.1):
    raw   = int(scale * math.sqrt(num_points))
    raw   = max(min_clusters, min(max_clusters, raw))
    power = 2 ** int(math.log2(raw))
    return power


class ClusteredVectorQuantizer(nn.Module):
    def __init__(self, num_clusters, codebook_dim=3, codebook_size=64,
                 num_quantizers=2, kmeans_iters=5):
        super().__init__()
        self.num_clusters   = num_clusters
        self.codebook_dim   = codebook_dim
        self.num_quantizers = num_quantizers

        self.sub_quantizers = nn.ModuleList([
            VectorQuantizer(
                num_quantizers=num_quantizers,
                codebook_dim=codebook_dim,
                codebook_size=codebook_size,
                kmeans_iters=kmeans_iters,
                vector_type="vector",
            )
            for _ in range(num_clusters)
        ])
        self.cluster_ids_ready = False

    @torch.no_grad()
    def build_clusters(self, feature_dc, scaling, rotation, n_init=10):
        feat    = build_cluster_features(
                      feature_dc.detach().cpu(),
                      scaling.detach().cpu(),
                      rotation.detach().cpu())
        feat_np = feat.numpy().astype(np.float32)

        km = KMeans(n_clusters=self.num_clusters, n_init=n_init,
                    max_iter=300, random_state=42)
        km.fit(feat_np)
        ids = torch.from_numpy(km.labels_).long()   # (N,)

        if self.cluster_ids_ready:
            self.cluster_ids.copy_(ids)
        else:
            self.register_buffer("cluster_ids", ids)
            self.cluster_ids_ready = True

        sizes = [(ids == c).sum().item() for c in range(self.num_clusters)]
        print(f"[ClusteredVQ] {self.num_clusters} clusters built, sizes = {sizes}")

    def forward(self, features, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.cluster_ids

        dequant    = torch.zeros_like(features)
        total_loss = features.new_zeros(1).squeeze()
        total_bits = 0

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            q, loss_c, bits_c = self.sub_quantizers[c](features[mask])
            dequant[mask] = q
            total_loss    = total_loss + loss_c
            total_bits   += bits_c

        return dequant, total_loss, total_bits

    def compress(self, features, cluster_ids=None):
        """
        Returns:
            dequant      : (N, C)
            cluster_ids  : (N,)
            index_tensor : (N, num_quantizers)  embed indices
        """
        if cluster_ids is None:
            cluster_ids = self.cluster_ids

        dequant      = torch.zeros_like(features)
        index_tensor = features.new_full(
            (features.shape[0], self.num_quantizers), -1).long()

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            q, idx = self.sub_quantizers[c].compress(features[mask])
            if idx.dim() == 1:
                idx = idx.unsqueeze(1)
            dequant[mask]                      = q
            index_tensor[mask, :idx.shape[1]] = idx

        return dequant, cluster_ids, index_tensor

    def decompress(self, cluster_ids, index_tensor, device):
        N     = cluster_ids.shape[0]
        recon = torch.zeros(N, self.codebook_dim, device=device)

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            idx         = index_tensor[mask]        # (n, num_quantizers)
            recon[mask] = self.sub_quantizers[c].decompress(idx)

        return recon

    def codebook_bits(self):
        bits = 0
        for sq in self.sub_quantizers:
            if sq.num_quantizers == 1:
                bits += sq.quantizer._codebook.embed.numel() * \
                        torch.finfo(sq.quantizer._codebook.embed.dtype).bits
            else:
                for layer in sq.quantizer.layers:
                    bits += layer._codebook.embed.numel() * \
                            torch.finfo(layer._codebook.embed.dtype).bits
        return bits


class ClusteredUniformQuantizer(nn.Module):
    def __init__(self, num_clusters, signed=False, bits=6, num_channels=2):
        super().__init__()
        self.num_clusters = num_clusters

        self.sub_quantizers = nn.ModuleList([
            UniformQuantizer(signed=signed, bits=bits,
                             learned=True, num_channels=num_channels)
            for _ in range(num_clusters)
        ])
        self.cluster_ids_ready = False

    def set_cluster_ids(self, cluster_ids):
        if self.cluster_ids_ready:
            self.cluster_ids.copy_(cluster_ids)
        else:
            self.register_buffer("cluster_ids", cluster_ids)
            self.cluster_ids_ready = True

    def _init_data(self, features, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.cluster_ids
        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            self.sub_quantizers[c]._init_data(features[mask])

    def forward(self, features, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.cluster_ids

        dequant    = torch.zeros_like(features)
        total_loss = features.new_zeros(1).squeeze()
        total_bits = 0

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            dq, loss_c, bits_c = self.sub_quantizers[c](features[mask])
            dequant[mask] = dq
            total_loss    = total_loss + loss_c
            total_bits   += bits_c

        return dequant, total_loss, total_bits

    def compress(self, features, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.cluster_ids

        quant_codes = torch.zeros_like(features)
        dequant     = torch.zeros_like(features)

        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            code_c, dq_c     = self.sub_quantizers[c].compress(features[mask])
            quant_codes[mask] = code_c
            dequant[mask]     = dq_c

        return quant_codes, dequant

    def decompress(self, quant_codes, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.cluster_ids

        dequant = torch.zeros_like(quant_codes)
        for c in range(self.num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() == 0:
                continue
            dequant[mask] = self.sub_quantizers[c].decompress(quant_codes[mask])
        return dequant

    def codebook_bits(self):
        bits = 0
        for sq in self.sub_quantizers:
            bits += sq.scale.numel() * torch.finfo(sq.scale.dtype).bits
            bits += sq.beta.numel()  * torch.finfo(sq.beta.dtype).bits
        return bits


class GaussianImage_RS_MultiBook(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type       = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W       = kwargs["H"], kwargs["W"]
        self.BLOCK_W         = kwargs["BLOCK_W"]
        self.BLOCK_H         = kwargs["BLOCK_H"]
        self.tile_bounds     = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]

        self._xyz         = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._scaling     = nn.Parameter(torch.rand(self.init_num_points, 2))
        self._rotation    = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.register_buffer('bound',    torch.tensor([0.5, 0.5]).view(1, 2))

        self.last_size           = (self.H, self.W)
        self.background          = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid
        self.quantize            = kwargs["quantize"]

        num_clusters = kwargs.get("num_clusters", None)
        if num_clusters is None:
            num_clusters = adaptive_num_clusters(self.init_num_points)
        self.num_clusters = num_clusters

        codebook_size     = kwargs.get("codebook_size",     64)
        vq_num_quantizers = kwargs.get("vq_num_quantizers",  2)

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply

            self.features_dc_quantizer = ClusteredVectorQuantizer(
                num_clusters   = num_clusters,
                codebook_dim   = 3,
                codebook_size  = codebook_size,
                num_quantizers = vq_num_quantizers,
                kmeans_iters   = 5,
            )
            self.scaling_quantizer = ClusteredUniformQuantizer(
                num_clusters  = num_clusters,
                signed        = False, bits=6, num_channels=2,
            )
            self.rotation_quantizer = ClusteredUniformQuantizer(
                num_clusters  = num_clusters,
                signed        = False, bits=6, num_channels=1,
            )

        # ---------- Optimizer ----------
        if kwargs.get("opt_type", "adam") == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5)

    @torch.no_grad()
    def build_clusters(self):
        print(f"[MultiBook] Building {self.num_clusters} clusters "
              f"from {self.init_num_points} Gaussians ...")

        self.features_dc_quantizer.build_clusters(
            feature_dc = self._features_dc.detach().cpu(),
            scaling    = self.get_scaling.detach().cpu(),
            rotation   = self.get_rotation.detach().cpu(),
        )
        # scaling / rotation 共享同一份 cluster_ids
        shared_ids = self.features_dc_quantizer.cluster_ids.cpu()
        self.scaling_quantizer.set_cluster_ids(shared_ids)
        self.rotation_quantizer.set_cluster_ids(shared_ids)

    def _init_data(self):
        """初始化各 cluster 的 UniformQuantizer scale/beta（build_clusters 之后调用）。"""
        if not self.features_dc_quantizer.cluster_ids_ready:
            self.build_clusters()
        ids = self.features_dc_quantizer.cluster_ids
        self.scaling_quantizer._init_data(self._scaling.detach(), ids)
        self.rotation_quantizer._init_data(self.get_rotation.detach(), ids)

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

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = \
            project_gaussians_2d_scale_rot(
                self.get_xyz, self.get_scaling, self.get_rotation,
                self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self.get_opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image      = render_pkg["render"]
        loss       = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            psnr = 10 * math.log10(1.0 / F.mse_loss(image, gt_image).item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def forward_quantize(self):
        ids = self.features_dc_quantizer.cluster_ids   # (N,) 固定不变

        m_bit  = 16 * self.init_num_points * 2
        means  = torch.tanh(self.xyz_quantizer(self._xyz))

        scaling,  l_vqs, s_bit = self.scaling_quantizer(self._scaling,     ids)
        scaling = torch.abs(scaling + self.bound)

        rotation, l_vqr, r_bit = self.rotation_quantizer(self.get_rotation, ids)
        colors,   l_vqc, c_bit = self.features_dc_quantizer(self.get_features, ids)

        self.xys, depths, self.radii, conics, num_tiles_hit = \
            project_gaussians_2d_scale_rot(
                means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        vq_loss = l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss,
                "unit_bit": [m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image      = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) \
             + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            psnr = 10 * math.log10(1.0 / F.mse_loss(image, gt_image).item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def compress_wo_ec(self):
        ids = self.features_dc_quantizer.cluster_ids

        quant_scaling,  _ = self.scaling_quantizer.compress(self._scaling.detach(), ids)
        quant_rotation, _ = self.rotation_quantizer.compress(self.get_rotation.detach(), ids)
        _, cluster_ids_out, dc_index_tensor = \
            self.features_dc_quantizer.compress(self.get_features.detach(), ids)

        return {
            "xyz":             self._xyz.half(),
            "cluster_ids":     cluster_ids_out,     # (N,)
            "quant_scaling":   quant_scaling,        # (N, 2)
            "quant_rotation":  quant_rotation,       # (N, 1)
            "dc_index_tensor": dc_index_tensor,      # (N, num_quantizers)
        }

    def decompress_wo_ec(self, enc):
        means    = torch.tanh(enc["xyz"].float())
        ids      = enc["cluster_ids"]

        scaling  = self.scaling_quantizer.decompress(enc["quant_scaling"],  ids)
        scaling  = torch.abs(scaling + self.bound)
        rotation = self.rotation_quantizer.decompress(enc["quant_rotation"], ids)
        colors   = self.features_dc_quantizer.decompress(
                       ids, enc["dc_index_tensor"], device=self.device)

        self.xys, depths, self.radii, conics, num_tiles_hit = \
            project_gaussians_2d_scale_rot(
                means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            colors, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def analysis_wo_ec(self, enc):
        qs_np  = enc["quant_scaling"].cpu().numpy()
        qr_np  = enc["quant_rotation"].cpu().numpy()
        dc_idx = enc["dc_index_tensor"].cpu().numpy()

        s_cb  = self.scaling_quantizer.codebook_bits()
        r_cb  = self.rotation_quantizer.codebook_bits()
        dc_cb = self.features_dc_quantizer.codebook_bits()

        dc_max_bit      = max(1, int(np.ceil(np.log2(dc_idx.max() + 2))))
        cluster_id_bits = self.init_num_points * max(1, int(math.log2(self.num_clusters)))

        position_bits   = self._xyz.numel() * 16
        scaling_bits    = s_cb  + qs_np.size * 6
        rotation_bits   = r_cb  + qr_np.size * 6
        feature_dc_bits = dc_cb + dc_idx.size * dc_max_bit

        total_bits = position_bits + scaling_bits + rotation_bits \
                   + feature_dc_bits + cluster_id_bits

        bpp = total_bits / self.H / self.W
        return {
            "bpp":            bpp,
            "position_bpp":   position_bits   / self.H / self.W,
            "scaling_bpp":    scaling_bits    / self.H / self.W,
            "rotation_bpp":   rotation_bits   / self.H / self.W,
            "cholesky_bpp":   (scaling_bits + rotation_bits) / self.H / self.W,
            "feature_dc_bpp": feature_dc_bits / self.H / self.W,
        }

    def compress(self):
        ids = self.features_dc_quantizer.cluster_ids

        scaling_index,  _ = self.scaling_quantizer.compress(self._scaling.detach(), ids)
        rotation_index, _ = self.rotation_quantizer.compress(self.get_rotation.detach(), ids)
        _, cluster_ids_out, dc_index_tensor = \
            self.features_dc_quantizer.compress(self.get_features.detach(), ids)

        return {
            "xyz":             self._xyz.half(),
            "cluster_ids":     cluster_ids_out,
            "scaling_index":   scaling_index,
            "rotation_index":  rotation_index,
            "dc_index_tensor": dc_index_tensor,
        }

    def decompress(self, enc):
        return self.decompress_wo_ec({
            "xyz":             enc["xyz"],
            "cluster_ids":     enc["cluster_ids"],
            "quant_scaling":   enc["scaling_index"],
            "quant_rotation":  enc["rotation_index"],
            "dc_index_tensor": enc["dc_index_tensor"],
        })

    def analysis(self, enc):
        """含熵编码 bit 统计：每个 cluster 单独熵编码，统计实际压缩大小。"""
        ids       = enc["cluster_ids"]
        s_idx     = enc["scaling_index"]
        r_idx     = enc["rotation_index"]
        dc_tensor = enc["dc_index_tensor"]

        s_ec, r_ec, dc_ec = 0, 0, 0
        s_hdr, r_hdr, dc_hdr = 0, 0, 0

        for c in range(self.num_clusters):
            mask = (ids == c)
            if mask.sum() == 0:
                continue

            # scaling
            comp, hist, uniq = compress_matrix_flatten_categorical(
                s_idx[mask].int().flatten().tolist())
            s_ec  += get_np_size(comp) * 8
            s_hdr += (get_np_size(hist) + get_np_size(uniq)) * 8

            # rotation
            comp, hist, uniq = compress_matrix_flatten_categorical(
                r_idx[mask].int().flatten().tolist())
            r_ec  += get_np_size(comp) * 8
            r_hdr += (get_np_size(hist) + get_np_size(uniq)) * 8

            # feature_dc embed indices
            comp, hist, uniq = compress_matrix_flatten_categorical(
                dc_tensor[mask].int().flatten().tolist())
            dc_ec  += get_np_size(comp) * 8
            dc_hdr += (get_np_size(hist) + get_np_size(uniq)) * 8

        s_cb  = self.scaling_quantizer.codebook_bits()
        r_cb  = self.rotation_quantizer.codebook_bits()
        dc_cb = self.features_dc_quantizer.codebook_bits()

        cluster_id_bits = self.init_num_points * max(1, int(math.log2(self.num_clusters)))
        position_bits   = self._xyz.numel() * 16
        scaling_bits    = s_cb  + s_hdr  + s_ec
        rotation_bits   = r_cb  + r_hdr  + r_ec
        feature_dc_bits = dc_cb + dc_hdr + dc_ec
        total_bits      = position_bits + scaling_bits + rotation_bits \
                        + feature_dc_bits + cluster_id_bits

        bpp = total_bits / self.H / self.W
        return {
            "bpp":            bpp,
            "position_bpp":   position_bits   / self.H / self.W,
            "scaling_bpp":    scaling_bits    / self.H / self.W,
            "rotation_bpp":   rotation_bits   / self.H / self.W,
            "cholesky_bpp":   (scaling_bits + rotation_bits) / self.H / self.W,
            "feature_dc_bpp": feature_dc_bits / self.H / self.W,
        }