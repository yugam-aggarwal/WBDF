from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Optional
from copy import copy

import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# ------------------------------- KDB utilities -------------------------------
# =============================================================================

def build_graph(X_train: np.ndarray, y_train: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
    """Learn a KDB structure: edges from y and up to k parents chosen by CMI.
    Returns list of (source, target) indices. y is the last index (n_features).
    """
    num_features = X_train.shape[1]
    x_nodes = list(range(num_features))
    y_node = num_features

    _x = lambda i: X_train[:, i]
    _x2comb = lambda i, j: (X_train[:, i], X_train[:, j])

    sorted_feature_idxs = np.argsort([
        drv.information_mutual(_x(i), y_train) for i in range(num_features)
    ])[::-1]

    edges: List[Tuple[int, int]] = []
    for it, target_idx in enumerate(sorted_feature_idxs):
        target_node = x_nodes[target_idx]
        edges.append((y_node, target_node))
        parent_candidate_idxs = sorted_feature_idxs[:it]
        if it <= k:
            for idx in parent_candidate_idxs:
                edges.append((x_nodes[idx], target_node))
        else:
            cmi_vals = [
                drv.information_mutual_conditional(*_x2comb(i, target_idx), y_train)
                for i in parent_candidate_idxs
            ]
            first_k_parent_mi_idxs = np.argsort(cmi_vals)[::-1][:k]
            first_k_parent_idxs = np.array(parent_candidate_idxs)[first_k_parent_mi_idxs]
            for parent_idx in first_k_parent_idxs:
                edges.append((x_nodes[parent_idx], target_node))
    return edges


def get_cross_table(*cols: np.ndarray, apply_wt: bool = False):
    if len(cols) == 0:
        raise TypeError("get_cross_table() requires at least one argument")
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")
    if not all(len(np.squeeze(col).shape) == 1 for col in cols):
        raise ValueError("all input arrays must be 1D")
    if apply_wt:
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = 1
    uniq_vals_all_cols, idx = zip(*(np.unique(col, return_inverse=True) for col in cols))
    shape_xt = [uv.size for uv in uniq_vals_all_cols]
    dtype_xt = "float" if apply_wt else "uint"
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt


def _get_dependencies_without_y(
    variables: Sequence[int], y_name: int, kdb_edges: Sequence[Tuple[int, int]]
) -> Dict[int, List[int]]:
    dependencies: Dict[int, List[int]] = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t: i for i, (s, t) in enumerate(kdb_edges) if s == y_name}
    for x in variables:
        current_dependencies = [s for s, t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            sort_dict = {t: mi_desc_order[t] for t in current_dependencies}
            dependencies[x] = sorted(sort_dict)
        else:
            dependencies[x] = current_dependencies
    return dependencies


class KdbHighOrderFeatureEncoder:
    """Encode per-feature high-order ids X_i | parents Π(i) as single integer ids, then one-hot.
    Stores constraints (group sizes) for BIN parameterization.
    """
    def __init__(self) -> None:
        self.dependencies_: Dict[int, List[int]] = {}
        self.constraints_: np.ndarray = np.array([])
        self.have_value_idxs_: List[np.ndarray] = []
        self.feature_uniques_: List[int] = []
        self.high_order_feature_uniques_: List[int] = []
        self.edges_: List[Tuple[int, int]] = []
        self.ohe: OneHotEncoder | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 3) -> "KdbHighOrderFeatureEncoder":
        edges = build_graph(X_train, y_train, k)
        num_features = X_train.shape[1]
        if k > 0:
            dependencies = _get_dependencies_without_y(list(range(num_features)), num_features, edges)
        else:
            dependencies = {x: [] for x in range(num_features)}
        self.dependencies_ = dependencies
        self.feature_uniques_ = [int(np.max(X_train[:, i]) + 1) for i in range(num_features)]
        self.edges_ = edges
        return self

    def _get_high_order_feature(self, X: np.ndarray, col: int, evidence_cols: Sequence[int]) -> np.ndarray:
        if not evidence_cols:
            return X[:, [col]]
        base = [1, self.feature_uniques_[col]] + [self.feature_uniques_[_col] for _col in evidence_cols[::-1][:-1]]
        cum_base = np.cumprod(base)[::-1]
        cols = list(evidence_cols) + [col]
        high_order_feature = np.sum(X[:, cols] * cum_base, axis=1).reshape(-1, 1)
        return high_order_feature

    def _get_high_order_constraints(self, X: np.ndarray, col: int, evidence_cols: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        if not evidence_cols:
            unique = self.feature_uniques_[col]
            return np.ones(unique, dtype=bool), np.array([unique])
        cols = list(evidence_cols) + [col]
        _, cross_table = get_cross_table(*[X[:, i] for i in cols])
        have_value = cross_table != 0
        have_value_reshape = have_value.reshape(-1, have_value.shape[-1])
        high_order_constraints = np.sum(have_value_reshape, axis=-1)
        return have_value, high_order_constraints

    def transform(self, X: np.ndarray, return_constraints: bool = True):
        high_order_features: List[np.ndarray] = []
        have_value_idxs: List[np.ndarray] = []
        constraints: List[np.ndarray] = []
        for k, v in self.dependencies_.items():
            hio = self._get_high_order_feature(X, k, v)
            idx, constraint = self._get_high_order_constraints(X, k, v)
            high_order_features.append(hio)
            have_value_idxs.append(idx)
            constraints.append(constraint)
        concated_constraints = np.hstack(constraints)
        concated_high_order_features = np.hstack(high_order_features)
        if self.ohe is None:
            self.ohe = OneHotEncoder(handle_unknown='ignore')
            self.ohe.fit(concated_high_order_features)
        X_high_order = self.ohe.transform(concated_high_order_features)
        # Only update constraints metadata when explicitly requested.
        # This ensures train-time constraints (used to size Broad layer)
        # are not overwritten by subsequent test-time transforms.
        if return_constraints:
            self.high_order_feature_uniques_ = [int(np.sum(c)) for c in constraints]
            self.constraints_ = concated_constraints
            self.have_value_idxs_ = have_value_idxs
            return X_high_order, concated_constraints, have_value_idxs
        else:
            return X_high_order

# =============================================================================
# ----------------------------- Wide φ₂ components ----------------------------
# =============================================================================

class WideCrossEncoder:
    """Compute φ₂(x): singletons + all pairwise crosses (i<j) as integer IDs per cross."""
    def __init__(self):
        self.pairs_: List[Tuple[int, int]] = []
        self.cards_: List[int] = []
        self.feature_uniques_: List[int] = []

    def fit(self, X: np.ndarray):
        n_feat = X.shape[1]
        self.feature_uniques_ = [int(np.max(X[:, i]) + 1) for i in range(n_feat)]
        pairs: List[Tuple[int, int]] = []
        cards: List[int] = []
        for i in range(n_feat):
            pairs.append((i, i))
            cards.append(self.feature_uniques_[i])
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                pairs.append((i, j))
                cards.append(self.feature_uniques_[i] * self.feature_uniques_[j])
        self.pairs_ = pairs
        self.cards_ = cards
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert len(self.pairs_) > 0
        out = np.zeros((X.shape[0], len(self.pairs_)), dtype=np.int64)
        for idx, (i, j) in enumerate(self.pairs_):
            if i == j:
                out[:, idx] = X[:, i]
            else:
                out[:, idx] = X[:, i] + X[:, j] * self.feature_uniques_[i]
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class WideLinearLayer(nn.Module):
    """Linear model over φ₂(x) using per-(i,j) embedding tables summed to class logits.
    Returns per-class probabilities via softmax.
    """
    def __init__(self, cards: Sequence[int], num_classes: int):
        super().__init__()
        self.cards = list(cards)
        self.num_classes = int(num_classes)
        self.embs = nn.ModuleList([
            nn.Embedding(c, self.num_classes) for c in self.cards
        ])
        for emb in self.embs:
            nn.init.xavier_normal_(emb.weight)
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x_ids: torch.LongTensor) -> torch.Tensor:
        parts = [emb(x_ids[:, col]) for col, emb in enumerate(self.embs)]  # [(B,C), ...]
        logits = torch.stack(parts, dim=0).sum(dim=0) + self.bias  # (B,C)
        return F.softmax(logits, dim=-1)

# =============================================================================
# ------------------------------- Broad (BIN) ---------------------------------
# =============================================================================

class BroadBINLayer(nn.Module):
    """BIN using Θ-probability constrained parameterization (Eq. 7).
    We keep an unconstrained weight matrix W_logits; for each group g of rows,
    we apply softmax over rows (dim=0) → Θ_g, then take log and use as weight rows.
    Input X should be one-hot (dense float) of shape (B, total_rows).
    """
    def __init__(self, group_sizes: Sequence[int], num_classes: int):
        super().__init__()
        self.group_sizes = list(map(int, group_sizes))
        self.total_rows = int(np.sum(self.group_sizes))
        self.num_classes = int(num_classes)
        self.W_logits = nn.Parameter(torch.zeros(self.total_rows, self.num_classes))
        nn.init.xavier_normal_(self.W_logits)
        self.bias = nn.Parameter(torch.zeros(self.num_classes))  # acts as log prior
        # Precompute start/end indices per group
        idxs = np.cumsum([0] + self.group_sizes)
        self.groups = [(int(idxs[i]), int(idxs[i+1])) for i in range(len(idxs)-1)]

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        # Build constrained weights
        W_rows: List[torch.Tensor] = []
        for (s, e) in self.groups:
            Wg = self.W_logits[s:e, :]  # (rows_g, C)
            Theta_g = F.softmax(Wg, dim=0)  # column-wise normalize over rows
            W_rows.append(torch.log(Theta_g + 1e-12))
        W = torch.cat(W_rows, dim=0)  # (total_rows, C)
        logits = x_onehot @ W + self.bias  # (B,C)
        return F.softmax(logits, dim=-1)

# =============================================================================
# ------------------------------ CIN (xDeepFM) --------------------------------
# =============================================================================

class CIN(nn.Module):
    """Compressed Interaction Network.
    X0: (B, F0, D), Xk: (B, Fk, D). For each layer l, compute
    H_l = sum_d ( X0[:,:,d] @ Xk[:,:,d]^T ) via vectorized outer interactions,
    then project to H_l feature maps and keep embedding dim D.
    If split_half: half features go to output concat; half propagate to next layer.
    Returns concatenated sum-pooled features over D: (B, sum(H_l)).
    """
    def __init__(self, field_dim: int, layer_sizes: Sequence[int], split_half: bool = True, activation: str = 'relu'):
        super().__init__()
        self.field_dim = int(field_dim)
        self.layer_sizes = list(layer_sizes)
        self.split_half = bool(split_half)
        self.act = getattr(F, activation) if activation != 'linear' else (lambda x: x)
        self.W = nn.ParameterList()  # per layer linear projection matrices
        self.fs = [self.field_dim]
        for i, h in enumerate(self.layer_sizes):
            # Linear projection from (F0 * Fk) → H
            self.W.append(nn.Parameter(torch.randn(self.field_dim * self.fs[-1], h)))
            nn.init.xavier_normal_(self.W[-1])
            # Next layer field dim
            next_f = h // 2 if (self.split_half and i != len(self.layer_sizes)-1) else h
            self.fs.append(next_f)

    def forward(self, X0: torch.Tensor, Xk: Optional[torch.Tensor] = None) -> torch.Tensor:
        # X0: (B, F0, D)
        if Xk is None:
            Xk = X0
        B, F0, D = X0.shape
        fk = Xk.shape[1]
        outputs: List[torch.Tensor] = []
        for li, h in enumerate(self.layer_sizes):
            # Outer interactions per embedding dimension
            # (B, F0, fk, D)
            outer = torch.einsum('bid,bjd->bijd', X0, Xk)
            # reshape to (B, D, F0*fk)
            outer = outer.permute(0, 3, 1, 2).contiguous().view(B, D, F0 * fk)
            # project to H feature maps independently for each d: (B, D, H)
            H = torch.matmul(outer, self.W[li])  # (B, D, H)
            H = self.act(H)
            # split
            if self.split_half and li != len(self.layer_sizes) - 1:
                H_out, H_next = torch.split(H, H.shape[-1] // 2, dim=-1)  # (B,D,H/2),(B,D,H/2)
            else:
                H_out, H_next = H, H
            # sum over embedding dim -> (B, H_out)
            outputs.append(H_out.sum(dim=1))
            # build next Xk: reshape (B, H_next, D)
            Xk = H_next.permute(0, 2, 1).contiguous()
            fk = Xk.shape[1]
        return torch.cat(outputs, dim=-1)

# =============================================================================
# ------------------------------ Deep (MLP head) -------------------------------
# =============================================================================

def make_activation(name: str) -> nn.Module:
    n = (name or "relu").lower()
    if n == "relu":        return nn.ReLU()
    if n == "gelu":        return nn.GELU()
    if n == "elu":         return nn.ELU()
    if n == "selu":        return nn.SELU()
    if n in ("leaky_relu", "lrelu"):  return nn.LeakyReLU(negative_slope=0.01)
    if n == "tanh":        return nn.Tanh()
    if n == "sigmoid":     return nn.Sigmoid()
    if n in ("linear","identity","none"): return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")

class DeepDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_units: Sequence[int], activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), make_activation(activation), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# =============================================================================
# ------------------------------ WBDF main model -------------------------------
# =============================================================================

class WBDFTorch(nn.Module):
    def __init__(
        self,
        *,
        vocab_sizes: Sequence[int],
        embed_dim: int,
        num_classes: int,
        use_wide: bool,
        wide_cards: Optional[Sequence[int]] = None,
        use_broad: bool = False,
        broad_group_sizes: Optional[Sequence[int]] = None,
        use_deep: bool = False,
        dnn_hidden: Sequence[int] = (256, 256, 256),
        use_cin: bool = False,
        cin_layer_sizes: Sequence[int] = (128, 128, 128),
        cin_split_half: bool = True,
    ):
        super().__init__()
        self.num_fields = len(vocab_sizes)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        # Embeddings shared by Deep and CIN
        self.embeddings = nn.ModuleList([nn.Embedding(v, self.embed_dim) for v in vocab_sizes])
        for emb in self.embeddings:
            nn.init.xavier_normal_(emb.weight)

        self.use_wide = use_wide
        self.use_broad = use_broad
        self.use_deep = use_deep
        self.use_cin = use_cin

        if self.use_wide:
            assert wide_cards is not None
            self.wide = WideLinearLayer(wide_cards, self.num_classes)
        if self.use_broad:
            assert broad_group_sizes is not None
            self.broad = BroadBINLayer(broad_group_sizes, self.num_classes)
        if self.use_deep:
            self.deep = DeepDNN(self.num_fields * self.embed_dim, dnn_hidden)
            self.deep_head = nn.Linear(dnn_hidden[-1], self.num_classes)
        if self.use_cin:
            self.cin = CIN(self.num_fields, cin_layer_sizes, split_half=cin_split_half, activation='relu')
            self.cin_head = nn.Linear(sum([s//2 if (cin_split_half and i != len(cin_layer_sizes)-1) else s
                                           for i, s in enumerate(cin_layer_sizes)]), self.num_classes)

        # Attention fusion: Dense(num_classes) -> ReLU -> Softmax
        # Input dim = (#enabled components) * num_classes
        comp_count = int(self.use_wide) + int(self.use_broad) + int(self.use_deep) + int(self.use_cin)
        self.attn = nn.Linear(comp_count * self.num_classes, self.num_classes)

    def _embed(self, x_cat: torch.LongTensor) -> torch.Tensor:
        # x_cat: (B, F)
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]  # [(B,D), ...]
        E = torch.stack(embs, dim=1)  # (B, F, D)
        return E

    def forward(
        self,
        *,
        x_cat: torch.LongTensor,                 # (B, F)
        wide_ids: Optional[torch.LongTensor],    # (B, n_pairs)
        broad_onehot: Optional[torch.FloatTensor]  # (B, total_rows)
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        E = self._embed(x_cat)  # (B,F,D)
        if self.use_deep:
            deep_feat = E.reshape(E.shape[0], -1)
            deep_logits = self.deep_head(self.deep(deep_feat))
            parts.append(F.softmax(deep_logits, dim=-1))
        if self.use_cin:
            cin_feat = self.cin(E)
            cin_logits = self.cin_head(cin_feat)
            parts.append(F.softmax(cin_logits, dim=-1))
        if self.use_wide:
            parts.append(self.wide(wide_ids))
        if self.use_broad:
            parts.append(self.broad(broad_onehot))
        # concat and attention
        z = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        y_logits = self.attn(z)
        y_relu = F.relu(y_logits)
        y_hat = F.softmax(y_relu, dim=-1)
        return y_hat

# =============================================================================
# ------------------------------- Data pipeline --------------------------------
# =============================================================================

class AdultDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, wide_ids: np.ndarray, broad_dense: np.ndarray, y_onehot: np.ndarray):
        self.x_cat = torch.from_numpy(x_cat.astype(np.int64))
        self.wide_ids = torch.from_numpy(wide_ids.astype(np.int64)) if wide_ids is not None else None
        self.broad = torch.from_numpy(broad_dense.astype(np.float32)) if broad_dense is not None else None
        self.y = torch.from_numpy(y_onehot.astype(np.float32))

    def __len__(self):
        return self.x_cat.shape[0]

    def __getitem__(self, idx):
        return {
            'x_cat': self.x_cat[idx],
            'wide_ids': None if self.wide_ids is None else self.wide_ids[idx],
            'broad': None if self.broad is None else self.broad[idx],
            'y': self.y[idx],
        }

# =============================================================================
# ---------------------------------- Demo run ----------------------------------
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(1024)
    np.random.seed(1024)

    DATA_PATH = "adult-dm.csv"
    data = pd.read_csv(DATA_PATH)

    target = data.columns[-1]
    feature_cols = [c for c in data.columns if c != target]

    # NOTE: Match Keras script semantics exactly
    # 1) Discretize/encode on the FULL dataset (before split)
    data[feature_cols] = data[feature_cols].fillna("-1")

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    if len(numeric_cols) > 0:
        kbin = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        data[numeric_cols] = kbin.fit_transform(data[numeric_cols]).astype('int32')

    for c in categorical_cols:
        data[c] = LabelEncoder().fit_transform(data[c].astype(str)).astype('int32')

    # Encode target on full data
    data[target] = LabelEncoder().fit_transform(data[target]).astype('int32')

    # 2) Train/test split (80/20)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Inputs for embeddings (Deep/CIN) and φ2
    X_train_cat = train[feature_cols].values.astype('int64')
    X_test_cat = test[feature_cols].values.astype('int64')

    # Vocab sizes computed from FULL data (to match Keras)
    vocab_sizes = [int(data[c].max() + 1) for c in feature_cols]

    # Labels one-hot
    num_classes = 2
    y_train_int = train[target].values.astype('int32')
    y_test_int = test[target].values.astype('int32')
    Y_train = np.eye(num_classes, dtype=np.float32)[y_train_int]
    Y_test = np.eye(num_classes, dtype=np.float32)[y_test_int]

    # Broad preprocessing on TRAIN ONLY (same as Keras code)
    enc = KdbHighOrderFeatureEncoder()
    enc.fit(X_train_cat, y_train_int, k=3)
    X_high_train, _, _ = enc.transform(X_train_cat, return_constraints=True)
    X_high_test = enc.transform(X_test_cat, return_constraints=False)
    X_high_train_np = X_high_train.toarray().astype('float32')
    X_high_test_np = X_high_test.toarray().astype('float32')

    # Wide φ2 preprocessing (fit on train; transform test)
    wx = WideCrossEncoder()
    wide_train_ids = wx.fit_transform(X_train_cat)
    wide_test_ids = wx.transform(X_test_cat)

    # Build full TRAIN dataset first
    full_train_ds = AdultDataset(X_train_cat, wide_train_ids, X_high_train_np, Y_train)

    # 3) validation_split=0.2 behavior: use the LAST 20% of TRAIN as validation (Keras default)
    n_train = len(full_train_ds)
    split_idx = int(np.ceil(n_train * 0.8))
    idx_train = list(range(0, split_idx))
    idx_val = list(range(split_idx, n_train))

    from torch.utils.data import Subset
    train_ds = Subset(full_train_ds, idx_train)
    val_ds = Subset(full_train_ds, idx_val)

    test_ds = AdultDataset(X_test_cat, wide_test_ids, X_high_test_np, Y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Model (same component toggles and dimensions)
    model = WBDFTorch(
        vocab_sizes=vocab_sizes,
        embed_dim=4,
        num_classes=num_classes,
        use_wide=True,
        wide_cards=wx.cards_,
        use_broad=True,
        broad_group_sizes=enc.constraints_.tolist(),
        use_deep=True,
        dnn_hidden=(256, 256, 256),
        use_cin=True,
        cin_layer_sizes=(128, 128, 128),
        cin_split_half=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def cat_ce_loss(probs: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        return -(targets_onehot * (probs + 1e-12).log()).sum(dim=1).mean()

    def run_epoch(loader: DataLoader, train_mode: bool = True):
        model.train(train_mode)
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            x_cat = batch['x_cat'].to(device)
            wide_ids = batch['wide_ids']
            broad = batch['broad']
            y = batch['y'].to(device)
            if wide_ids is not None:
                wide_ids = wide_ids.to(device)
            if broad is not None:
                broad = broad.to(device)
            with torch.set_grad_enabled(train_mode):
                y_hat = model(x_cat=x_cat, wide_ids=wide_ids, broad_onehot=broad)
                loss = cat_ce_loss(y_hat, y)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item()) * x_cat.size(0)
            pred = y_hat.argmax(dim=1)
            targ = y.argmax(dim=1)
            correct += int((pred == targ).sum().item())
            total += int(x_cat.size(0))
        return total_loss / total, correct / total

    # Train (report val like Keras fit with validation_split)
    for epoch in range(1, 2):
        train_loss, train_acc = run_epoch(train_loader, True)
        val_loss, val_acc = run_epoch(val_loader, False)
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    # Final test evaluation (matches model.evaluate on test)
    test_loss, test_acc = run_epoch(test_loader, False)
    print({"loss": float(test_loss), "acc": float(test_acc)})
