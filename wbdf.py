# %%
import math, random, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml

# Repro
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %%
def load_dataset(dataset_name='iris'):
    """
    Load a dataset by name
    Returns:
        X_df: DataFrame with features
        y_arr: Array with target values
        num_cols: List of numeric column names
        cat_cols: List of categorical column names
    """
    dataset_name = dataset_name.lower()
    # Built-in datasets from sklearn
    if dataset_name == 'iris':
        data = load_iris(as_frame=True)
        X_df = data.data; y_arr = data.target
    elif dataset_name == 'wine':
        data = load_wine(as_frame=True)
        X_df = data.data; y_arr = data.target
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer(as_frame=True)
        X_df = data.data; y_arr = data.target
    # Fetch from OpenML
    elif dataset_name == 'adult':
        data = fetch_openml(name='adult', version=2, as_frame=True)
        X_df = data.data; y_arr = LabelEncoder().fit_transform(data.target)
    elif dataset_name == 'titanic':
        data = fetch_openml(name='titanic', version=1, as_frame=True)
        X_df = data.data; y_arr = LabelEncoder().fit_transform(data.target)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

    # Determine column types automatically
    num_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Convert boolean columns to string for preprocessing
    for col in X_df.select_dtypes(include=['bool']).columns:
        X_df[col] = X_df[col].astype(str)

    if hasattr(y_arr, 'values'):
        y_arr = y_arr.values

    print(f"Loaded {dataset_name} dataset with {X_df.shape[0]} samples and {X_df.shape[1]} features")
    print(f"- Numeric features: {len(num_cols)}")
    print(f"- Categorical features: {len(cat_cols)}")
    print(f"- Target classes: {len(np.unique(y_arr))}")
    return X_df, y_arr, num_cols, cat_cols

# Choose dataset here
dataset_name = 'titanic'  # options: 'iris', 'wine', 'breast_cancer', 'adult', 'titanic'
X_df, y_arr, num_cols, cat_cols = load_dataset(dataset_name)

# --- preprocessing pipeline: discretize numerics -> OHE everything ---
sk_ver = tuple(map(int, __import__("sklearn").__version__.split(".")[:2]))
ohe_kwargs = {"handle_unknown": "ignore"}
if sk_ver >= (1, 2):
    ohe_kwargs["sparse_output"] = True
else:
    ohe_kwargs["sparse"] = True

# Determine appropriate number of bins based on dataset size
n_samples = len(X_df)
n_bins = 5 if n_samples < 1000 else 10  # fewer bins for small datasets

# Only create preprocessing steps for column types that exist
transformers = []
if num_cols:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("bin", KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile"))
    ])
    transformers.append(("num", numeric, num_cols))
if cat_cols:
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent"))
    ])
    transformers.append(("cat", categorical, cat_cols))

coltf = ColumnTransformer(transformers=transformers, remainder="drop")
pipe = Pipeline([("coltf", coltf), ("ohe", OneHotEncoder(**ohe_kwargs))])

X_all = pipe.fit_transform(X_df)  # sparse CSR
num_classes = len(np.unique(y_arr))

# Train/test split for tensors (OHE)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_arr, test_size=0.2, stratify=y_arr, random_state=SEED
)
if hasattr(y_train, 'values'): y_train = y_train.values
if hasattr(y_test, 'values'): y_test = y_test.values

X_train = torch.from_numpy(X_train.toarray()).float()
X_test  = torch.from_numpy(X_test.toarray()).float()
y_train = torch.from_numpy(y_train).long()
y_test  = torch.from_numpy(y_test).long()

print("Input dim after OHE:", X_train.shape[1], "| Train size:", X_train.shape, "| Test size:", X_test.shape)

# --- Pre-OHE discrete matrix + OHE field slices (for Broad KDB-k) ---
ohe = pipe.named_steps["ohe"]
coltf_only = pipe.named_steps["coltf"]

# Pre-OHE columns = original fields (numerics binned, categoricals passthrough)
X_pre = coltf_only.transform(X_df)
field_names = (num_cols or []) + (cat_cols or [])
F = len(field_names)

# Recompute split indices identically for Broad's Θ (same seed/stratify)
idx_all = np.arange(len(X_df))
train_idx, test_idx = train_test_split(idx_all, test_size=0.2, stratify=y_arr, random_state=SEED)

# Convert each pre-OHE column to integer codes aligned with ohe.categories_[f]
codes = np.zeros((len(X_df), F), dtype=np.int64)
for f in range(F):
    cats = np.array(ohe.categories_[f], dtype=object)
    col = X_pre[:, f]
    
    # Handle NaN in categories: filter them out and create a mapping
    valid_mask = pd.notna(cats)
    valid_cats = cats[valid_mask]
    
    if len(valid_cats) == 0:
        # All categories are NaN, assign all to 0
        codes[:, f] = 0
        continue
    
    # Create categorical with valid categories only
    cat_series = pd.Categorical(pd.Series(col).where(pd.notna(col), None), categories=valid_cats)
    codes[:, f] = cat_series.codes
    # Unknown or NaN values get -1 from pd.Categorical, map to 0
    codes[:, f] = np.where(codes[:, f] < 0, 0, codes[:, f])

codes_train = codes[train_idx]
y_train_np  = y_arr[train_idx]
codes_test  = codes[test_idx]
y_test_np   = y_arr[test_idx]

# OHE field slices (start:end) for each original field
cats_lens = [len(c) for c in ohe.categories_]
offsets = np.cumsum([0] + cats_lens)
field_slices = [(offsets[i], offsets[i+1]) for i in range(F)]

# %%
# Models: Wide (quadratic), Deep (MLP), optional CIN (not used by default here)
in_dim = X_train.shape[1]

class Wide(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        quad_dim = in_dim + (in_dim * (in_dim + 1)) // 2
        self.linear = nn.Linear(quad_dim, num_classes)
    def compute_quadratic_features(self, x):
        B, p = x.shape
        crosses = []
        for i in range(p):
            for j in range(i, p):
                crosses.append((x[:, i] * x[:, j]).unsqueeze(1))
        cross_terms = torch.cat(crosses, dim=1)
        return torch.cat([x, cross_terms], dim=1)
    def forward(self, x):
        x_tilde = self.compute_quadratic_features(x)
        return self.linear(x_tilde)  # logits

class Deep(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=(256,128)):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.2)]
            d=h
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)  # logits

class CIN(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, cin_embed_dim: int = 16,
                 cin_layer_sizes: tuple = (64, 64), cin_split_half: bool = True,
                 cin_activation: str = "relu"):
        super().__init__()
        assert in_dim > 0
        assert len(cin_layer_sizes) > 0
        self.m = in_dim; self.d = cin_embed_dim
        self.layer_sizes = list(cin_layer_sizes); self.split_half = cin_split_half
        self.field_embedding = nn.Parameter(torch.empty(self.m, self.d))
        nn.init.normal_(self.field_embedding, mean=0.0, std=0.02)
        convs=[]; prev_H = self.m
        for k, Hk in enumerate(self.layer_sizes):
            conv = nn.Conv1d(prev_H*self.m, Hk, kernel_size=1, bias=True)
            nn.init.xavier_uniform_(conv.weight); nn.init.zeros_(conv.bias)
            convs.append(conv); prev_H = (Hk//2 if self.split_half and k < len(self.layer_sizes)-1 else Hk)
        self.convs = nn.ModuleList(convs)
        self.act = nn.Identity() if (cin_activation is None or cin_activation.lower()=="identity") \
                   else (nn.ReLU() if cin_activation.lower()=="relu" else getattr(nn, cin_activation)())
        out_width = 0
        for k, Hk in enumerate(self.layer_sizes):
            out_width += (Hk - (Hk//2) if self.split_half and k < len(self.layer_sizes)-1 else Hk)
        self.fc = nn.Linear(out_width, num_classes)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        X0 = torch.einsum('bi,ij->bij', x, self.field_embedding)  # [B,m,d]
        Xk_prev = X0; prev_H = self.m; outs=[]
        for li, (conv, Hk) in enumerate(zip(self.convs, self.layer_sizes)):
            Zk = torch.einsum('bhd,bmd->bhmd', Xk_prev, X0).reshape(B, prev_H*self.m, self.d)
            Xk = self.act(conv(Zk))
            is_last = (li == len(self.layer_sizes)-1)
            if self.split_half and not is_last:
                Hn = Hk//2; outs.append(Xk[:, Hn:, :]); Xk_prev = Xk[:, :Hn, :]; prev_H = Hn
            else:
                outs.append(Xk); Xk_prev = Xk; prev_H = Hk
        cin_out = torch.cat([t.sum(dim=2) for t in outs], dim=1)
        return self.fc(cin_out)

# %%
# -------- Broad (BIN) with KDB-k structure --------
class Broad(nn.Module):
    """
    Broad/BIN with KDB-k structure:
      - Structure: each Xi depends on Y and up to k feature-parents chosen by I(Xi; Xj | Y),
                   with features ordered by I(X;Y).
      - Parameters: Θ (Laplace-smoothed CPTs) + trainable B (weights on log Θ) + w_y on log P(Y).
      - Output logits: w_y * log P(Y) + Σ_i  B_i[y, parent_combo, xi] * log P(Xi | Y, parents_i)
    """
    def __init__(self, ohe, field_slices, train_codes, train_y, num_classes,
                 structure="kdb", k=1, alpha=1.0):
        super().__init__()
        assert structure.lower() in {"kdb", "nb"}
        self.ohe = ohe
        self.field_slices = field_slices
        self.num_fields = len(field_slices)
        self.num_classes = int(num_classes)
        self.cardinals = [len(c) for c in ohe.categories_]
        self.alpha = float(alpha)
        self.k = int(k)

        # 1) Learn KDB-k parents (NB if k <= 0)
        if structure.lower() == "nb" or self.k <= 0:
            self.parents = [[] for _ in range(self.num_fields)]
        else:
            self.parents = self._learn_kdb_parents(train_codes, train_y, self.cardinals, self.num_classes, self.k)

        # 2) Estimate Θ with Laplace smoothing
        self.theta_y, self.theta_tables, self.parent_radixes = self._estimate_thetas(train_codes, train_y)

        # Cache log Θ as tensors
        self.log_theta_y = torch.from_numpy(np.log(self.theta_y).astype(np.float32))
        self.log_theta_tables = [torch.from_numpy(np.log(t).astype(np.float32)) for t in self.theta_tables]

        # 3) Discriminative weights
        self.w_y = nn.Parameter(torch.ones(self.num_classes))
        self.B_params = nn.ParameterList()
        for i in range(self.num_fields):
            Pcard = self.theta_tables[i].shape[1]
            Vi = self.cardinals[i]
            self.B_params.append(nn.Parameter(torch.ones(self.num_classes, Pcard, Vi)))

    # ---- MI(X;Y) ----
    @staticmethod
    def _mi_xy(codes, y, i, Ci, C):
        N = len(y)
        M = np.zeros((Ci, C), dtype=np.float64)
        np.add.at(M, (codes[:, i], y), 1.0)
        Pxy = M / N; Px = Pxy.sum(axis=1, keepdims=True); Py = Pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = Px * Py
            ratio = np.where(denom > 0, Pxy / denom, 1.0)
            return float(np.where(Pxy > 0, Pxy * np.log(ratio), 0.0).sum())

    # ---- CMI(Xi;Xj|Y) ----
    @staticmethod
    def _cmi_pair(codes, y, i, j, Ci, Cj, C):
        N = len(y); cmi = 0.0
        for k in range(C):
            mask = (y == k); Nk = int(mask.sum())
            if Nk == 0: continue
            xi = codes[mask, i]; xj = codes[mask, j]
            M = np.zeros((Ci, Cj), dtype=np.float64); np.add.at(M, (xi, xj), 1.0)
            Pxy = M / Nk; Px = Pxy.sum(axis=1, keepdims=True); Py = Pxy.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = Px * Py
                ratio = np.where(denom > 0, Pxy / denom, 1.0)
                term = np.where(Pxy > 0, Pxy * np.log(ratio), 0.0).sum()
            cmi += (Nk / N) * term
        return float(cmi)

    # ---- learn KDB-k parents ----
    def _learn_kdb_parents(self, codes, y, cardinals, C, k):
        F = codes.shape[1]
        # 1) order by MI(X;Y) desc
        mis = [self._mi_xy(codes, y, i, cardinals[i], C) for i in range(F)]
        order = list(np.argsort(mis))[::-1]
        # 2) choose up to k parents among earlier features by top CMI
        parents = [[] for _ in range(F)]
        for idx, i in enumerate(order):
            candidates = order[:idx]
            if not candidates or k <= 0: 
                continue
            scores = [(j, self._cmi_pair(codes, y, i, j, cardinals[i], cardinals[j], C)) for j in candidates]
            scores.sort(key=lambda t: t[1], reverse=True)
            parents[i] = [j for j,_ in scores[:k]]
        return parents

    # ---- estimate Θ with Laplace smoothing ----
    def _estimate_thetas(self, codes, y):
        N = len(y); F = codes.shape[1]; C = self.num_classes; A = self.alpha
        # P(Y)
        cls = np.full(C, A, dtype=np.float64); np.add.at(cls, y, 1.0)
        theta_y = cls / cls.sum()

        theta_tables = []; parent_radixes = []
        for i in range(F):
            Vi = self.cardinals[i]; ps = self.parents[i]
            if len(ps) == 0:
                Pcard = 1; radixes = np.array([], dtype=np.int64)
                parent_idx = np.zeros(N, dtype=np.int64)
            else:
                par_cards = [self.cardinals[p] for p in ps]
                radixes = np.concatenate(([1], np.cumprod(par_cards[:-1], dtype=np.int64)))  # len=r
                parent_idx = (codes[:, ps] * radixes).sum(axis=1)
                Pcard = int(np.prod(par_cards))
            counts = np.full((C, Pcard, Vi), A, dtype=np.float64)
            np.add.at(counts, (y, parent_idx, codes[:, i]), 1.0)
            denom = counts.sum(axis=2, keepdims=True)
            theta = counts / np.maximum(denom, 1e-12)
            theta_tables.append(theta); parent_radixes.append(radixes)
        return theta_y, theta_tables, parent_radixes

    # ---- forward: gather indices and score ----
    def forward(self, x_dense):
        device = x_dense.device; Bsz = x_dense.size(0); C = self.num_classes
        log_theta_y = self.log_theta_y.to(device)  # [C]
        logits = (self.w_y.to(device) * log_theta_y).unsqueeze(0).expand(Bsz, C)

        for i in range(self.num_fields):
            s, e = self.field_slices[i]
            xi_val = x_dense[:, s:e].argmax(dim=1)  # [B]
            ps = self.parents[i]
            if len(ps) == 0:
                parent_idx = torch.zeros(Bsz, dtype=torch.long, device=device)
            else:
                par_vals = []
                for p in ps:
                    ps_, pe_ = self.field_slices[p]
                    par_vals.append(x_dense[:, ps_:pe_].argmax(dim=1))
                par_vals = torch.stack(par_vals, dim=1)  # [B, r]
                radixes = torch.as_tensor(self.parent_radixes[i], device=device, dtype=torch.long)  # [r]
                parent_idx = (par_vals * radixes).sum(dim=1)

            log_tab = self.log_theta_tables[i].to(device)  # [C, Pcard, Vi]
            B_tab   = self.B_params[i].to(device)          # [C, Pcard, Vi]
            log_sel = log_tab.permute(1, 2, 0)[parent_idx, xi_val]  # [B, C]
            B_sel   = B_tab.permute(1, 2, 0)[parent_idx, xi_val]    # [B, C]
            # Replace in-place addition with regular addition
            logits = logits + B_sel * log_sel
        return logits

# %%
class Gate(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 32, nhead: int = 4, num_components: int = 4):
        super().__init__()
        self.token_proj = nn.Linear(num_classes, d_model)
        self.type_embed = nn.Parameter(torch.zeros(num_components, d_model))
        nn.init.normal_(self.type_embed, std=0.02)
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model * num_components),
            nn.Linear(d_model * num_components, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, component_logits):
        x = torch.stack(component_logits, dim=1)  # [B, num_active, C]
        x = self.token_proj(x) + self.type_embed[:len(component_logits)].unsqueeze(0)
        attn_out, _ = self.mha(x, x, x)
        concatenated = attn_out.flatten(1)
        return self.ff(concatenated)

class WBDF(nn.Module):
    def __init__(self, in_dim, num_classes=2, deep_hidden=(256,128), d_model=32, nhead=4,
                 use_wide=True, use_deep=True, use_factorized=True, use_broad=True,
                 broad_kwargs=None):
        super().__init__()
        self.use_wide = use_wide
        self.use_deep = use_deep
        self.use_factorized = use_factorized
        self.use_broad = use_broad

        active_count = sum([use_wide, use_deep, use_factorized, use_broad])
        if active_count == 0:
            raise ValueError("At least one component must be enabled")

        if use_wide:
            self.wide = Wide(in_dim, num_classes)
        if use_deep:
            self.deep = Deep(in_dim, num_classes, hidden=deep_hidden)
        if use_factorized:
            self.factorized = CIN(in_dim, num_classes)
        if use_broad:
            if broad_kwargs is None:
                raise ValueError("broad_kwargs required when use_broad=True")
            self.broad = Broad(**broad_kwargs)

        self.combine = Gate(num_classes, d_model=d_model, nhead=nhead, num_components=active_count)

    def forward(self, x):
        component_logits = []
        if self.use_wide: component_logits.append(self.wide(x))
        if self.use_deep: component_logits.append(self.deep(x))
        if self.use_factorized: component_logits.append(self.factorized(x))
        if self.use_broad: component_logits.append(self.broad(x))
        return self.combine(component_logits)   # [B, C] logits

# %%
# ==========================
# Step 4 — Train & evaluate
# ==========================
batch_size = 256
epochs = 8
lr = 1e-3

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, drop_last=False)

# Broad (KDB-k) args
broad_kwargs = dict(
    ohe=ohe,
    field_slices=field_slices,
    train_codes=codes_train,
    train_y=y_train_np,
    num_classes=num_classes,
    structure="kdb",   # 'kdb' or 'nb'
    k=1,               # expose k (parent cap). Start with k=1.
    alpha=1.0          # Laplace smoothing
)

in_dim = X_train.shape[1]
model = WBDF(in_dim,
             num_classes=num_classes,
             use_wide=False,        # keep off for speed on large OHE
             use_deep=True,
             use_factorized=False,  # off for speed; you can enable later
             use_broad=True,
             broad_kwargs=broad_kwargs).to(device)

opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()  # expects logits

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total

for epoch in range(1, epochs+1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item() * yb.size(0)
    train_loss = running / len(train_ds)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | test_acc={acc:.3f}")
