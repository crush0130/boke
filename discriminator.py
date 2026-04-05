import torch
from torch import nn
from torch.nn.utils import spectral_norm

import torch.nn.functional as F

import numpy as np
import copy
import math
from foldingdiff.self_attention import gather_edges, gather_nodes, Normalize, cat_neighbors_nodes, EncLayer




class TransformerPositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(TransformerPositionEncoding, self).__init__()

        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)
        half_dim = d_model // 2
        ## emb.shape (hald_dim, )
        emb = torch.exp(torch.arange(half_dim) * -(math.log(10000) / half_dim))
        # Compute the positional encodings once in log space.
        pe[: ,: half_dim] = torch.sin(position[:, None] * emb)
        pe[: ,half_dim: ] = torch.cos(position[:, None] * emb)

        self.register_buffer("pe", pe, persistent=True)

    def forward(self, timesteps, index_select=False):
        """
        return [:seqlen, d_model]
        """
        if not index_select:
            assert len(timesteps.shape) == 1
            return self.pe[:timesteps.shape[0]]
        else:
            timesteps = timesteps.long()
            # 处理三维输入 [batch_size, res_num, knn]
            if len(timesteps.shape) == 3:
                B, L, K = timesteps.shape
                # 展平后再重塑
                encoded = self.pe[timesteps.reshape(-1)].reshape(B, L, K, self.d_model)
                return encoded
            else:
                # 原有的二维处理逻辑
                B, L = timesteps.shape
                return self.pe[timesteps.reshape(-1)].reshape(B, L, self.d_model)


def moveaxis(data, source, destination):
  n_dims = len(data.shape)
  dims = [i for i in range(n_dims)]
  if source < 0:
    source += n_dims
  if destination < 0:
    destination += n_dims

  if source < destination:
    dims.pop(source)
    dims.insert(destination, source)
  else:
    dims.pop(source)
    dims.insert(destination, source)

  return data.permute(*dims)


class ProteinMPNNFeaturesNew(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, features_type='mpnn', augment_eps=0., dropout=0.1, max_len=500000,
                 node_angle_len=7):
        """ Extract protein features """
        super(ProteinMPNNFeaturesNew, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.node_angle_len = node_angle_len
        self.node_pad_num = self.node_angle_len // 2

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'mpnn': (6 * node_angle_len, num_rbf * 16, 7)
        }

        # Positional encoding
        # self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.embeddings = TransformerPositionEncoding(max_len, node_features)
        self.dropout = nn.Dropout(dropout)

        # Normalization and embedding
        node_in, edge_dist_in, edge_orient_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_dist_embedding = nn.Linear(edge_dist_in, edge_features, bias=True)
        self.edge_orient_embedding = nn.Linear(edge_orient_in, edge_features, bias=True)

        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        E_idx = E_idx.long()
        return D_neighbors, E_idx

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

        return RBF

    def _dihedrals(self, X, eps=1e-7):
        res_num = X.shape[1]
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        # phi, psi, omega = torch.unbind(D,-1)

        pad_D = F.pad(D, (0, 0, self.node_pad_num, self.node_pad_num), 'constant', 0)
        D_angles = torch.stack([
            pad_D.transpose(1, 0)[torch.arange(self.node_angle_len) + node_idx].transpose(1, 0).reshape(-1,
                                                                                                        self.node_angle_len * 3)
            for node_idx in np.arange(res_num)], 1)
        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D_angles), torch.sin(D_angles)), 2)
        return D_features

    def rot_to_quat(self, rot, unstack_inputs=False):

        if unstack_inputs:
            rot = [moveaxis(x, -1, 0) for x in moveaxis(rot, -2, 0)]

        [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

        # pylint: disable=bad-whitespace
        k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy, ],
             [zy - yz, xx - yy - zz, xy + yx, xz + zx, ],
             [xz - zx, xy + yx, yy - xx - zz, yz + zy, ],
             [yx - xy, xz + zx, yz + zy, zz - xx - yy, ]]
        # pylint: enable=bad-whitespace

        k = (1. / 3.) * torch.stack([torch.stack(x, dim=-1) for x in k],
                                    dim=-2)

        # Get eigenvalues in non-decreasing order and associated.
        # _, qs = torch.linalg.eigh(k)
        kk = np.array(k.detach().cpu().numpy(), dtype=np.float32)
        # import pdb; pdb.set_trace()
        _, qss = np.linalg.eigh(kk)
        # try:
        #   _, qss = np.linalg.eigh(kk)
        # except:
        #   import pdb; pdb.set_trace()
        qs = torch.from_numpy(qss).to(k.device)
        return qs[..., -1]

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(1e-10 + torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True)) + 1e-10) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # import pdb; pdb.set_trace()
        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        # import pdb; pdb.set_trace()
        Q = self.rot_to_quat(rot=R, unstack_inputs=True)
        # Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)

        return O_features

    def forward(self, X, L, mask, single_res_rel):
        X = X.float()
        mask = mask.float()
        single_res_rel = single_res_rel.float()

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        # O = X[:,:,3,:]

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        D_neighbors, E_idx = self._dist(X_ca, mask)
        # RBF = self._rbf(D_neighbors)

        # Pairwise features
        # import pdb; pdb.set_trace()
        O_features = self._orientations_coarse(X_ca, E_idx)
        # O_features = self._orientations_frame(X, E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        # RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        # RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        # RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        # RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        # RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        # RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        # RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        # RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        # RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # Pairwise embeddings
        # E_positional = self.embeddings(E_idx)
        batch_size, res_num, knn = E_idx.shape
        E_single_res_rel = torch.gather(single_res_rel, dim=2, index=E_idx)
        E_positional = self.embeddings(E_single_res_rel, index_select=True).reshape(batch_size, res_num, knn, -1)
        # Full backbone angles
        V = self._dihedrals(X)
        V = V.float()
        # E = torch.cat((E_positional, RBF_all, O_features), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_dist_embedding(RBF_all) + self.edge_orient_embedding(O_features) + E_positional
        E = self.norm_edges(E)

        return V, E, E_idx

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B


class MPNNEncoder(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim,
                 num_encoder_layers=3, vocab=22, k_neighbors=30,
                 protein_features='mpnn', augment_eps=0., dropout=0.1):
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinMPNNFeaturesNew(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        # self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, L, mask, single_res_rel):
        """ Graph-conditioned sequence model """
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask, single_res_rel)
        # h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        hidden_list = []
        for layer in self.encoder_layers:
            # 保存输入用于残差
            h_V_input = h_V
            h_E_input = h_E
            # 编码器层处理
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            # 层间残差链接
            h_V = h_V + h_V_input
            h_E = h_E + h_E_input

            h_VE_encoder = cat_neighbors_nodes(h_V, h_E, E_idx)
            hidden_list.append(h_VE_encoder)

        feature_dict = {
            'out_feature': h_VE_encoder,
            'stacked_hidden': torch.stack(hidden_list)
        }

        return feature_dict


class LocalEnvironmentTransformer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # === Backbone: message passing over local geometry ===
        self.structuretransformer = MPNNEncoder(
            node_features=config.node_features,
            edge_features=config.edge_features,
            hidden_dim=config.hidden_dim,
            num_encoder_layers=config.num_encoder_layers,
            k_neighbors=config.k_neighbors,
        )

        # === Critic head (ONLY ONE linear layer) ===
        # 关键 1️⃣：spectral norm 保证近似 1-Lipschitz
        self.LE_out_projection = spectral_norm(
            nn.Linear(config.hidden_dim, 1)
        )

        # 关键 2️⃣：输出缩放，防止 critic 突然“觉醒”
        self.critic_scale = getattr(config, "critic_scale", 5.0)

    # ------------------------------------------------------------------
    # Main forward: returns real / fake scores + hidden features
    # ------------------------------------------------------------------
    def forward(self, batch, pred_dict, detach_all=False):
        """
        Used during training.
        """

        # ===== Real structure =====
        true_node_hidden, true_logits = self.process(
            batch["coords"],
            batch["attn_mask"],
            batch["single_res_rel"],
        )

        # ===== Predicted structure =====
        pred_coord = pred_dict["all_coord"][-1]
        if detach_all:
            pred_coord = pred_coord.detach()

        pred_node_hidden, pred_logits = self.process(
            pred_coord,
            batch["attn_mask"],
            batch["single_res_rel"],
        )

        return true_logits, pred_logits, true_node_hidden, pred_node_hidden

    # ------------------------------------------------------------------
    # Core processing logic
    # ------------------------------------------------------------------
    def process(self, coords, attn_mask, single_res_rel):

        device = attn_mask.device
        B, L = attn_mask.shape

        # ===== Normalize coordinate shape =====
        # [B, L, A, 3]
        if coords.dim() == 4:
            pass

        # [B, L*A, 3]
        elif coords.dim() == 3:
            if coords.shape[0] == B:
                num_atoms = coords.shape[1] // L
                coords = coords.view(B, L, num_atoms, 3)
            else:
                raise ValueError(
                    f"coords shape {coords.shape} incompatible with B={B}, L={L}"
                )

        # [L*A, 3]  -> assume B=1
        elif coords.dim() == 2:
            num_atoms = coords.shape[0] // L
            coords = coords.view(1, L, num_atoms, 3)
            attn_mask = attn_mask[:1]
            single_res_rel = single_res_rel[:1]

        else:
            raise ValueError(f"Unexpected coords shape {coords.shape}")

        # ===== Graph feature extraction =====
        V, E, E_idx = self.structuretransformer.features(
            coords, L, attn_mask, single_res_rel
        )

        h_V = self.structuretransformer.W_v(V)  # [B, L, hidden]
        h_E = self.structuretransformer.W_e(E)  # unused but kept for completeness

        node_hidden = h_V

        # ===== Critic score per residue =====
        single_score = self.LE_out_projection(node_hidden).squeeze(-1)  # [B, L]

        # 关键 3️⃣：tanh + scale → soft bounded critic
        single_score = torch.tanh(single_score / self.critic_scale)

        # ===== Masked sequence-level aggregation =====
        mask = attn_mask.float()
        logits = (single_score * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return node_hidden, logits

    # ------------------------------------------------------------------
    # Optional: inference-only forward (not used in WGAN training)
    # ------------------------------------------------------------------
    def forward_coords(self, coords, attn_mask, single_res_rel):
        node_hidden, logits = self.process(coords, attn_mask, single_res_rel)
        return logits