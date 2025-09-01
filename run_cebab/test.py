# table2_runner.py
import os, re, random, json, math, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import (BertTokenizer, BertModel,
                          DistilBertTokenizer, DistilBertModel,
                          RobertaTokenizer, RobertaModel)
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ======= 填好 IMDB-C 的 CSV 路径（四列概念 + 文本 + 标签），列名见下方 loader =======
IMDBC_TRAIN = "./imdbc/train.csv"
IMDBC_DEV   = "./imdbc/dev.csv"
IMDBC_TEST  = "./imdbc/test.csv"
# 预期列： text, label (0/1), acting, storyline, emotion, cinematography （概念值取 {0:negative,1:positive,2:unknown}）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 16
EPOCHS = 5
LR     = 2e-5
LAMBDA_X2C = 5.0  # 论文 joint 训练默认
MAXLEN = 128

# ========== 轻量模型映射 ==========
BACKBONES = {
    "DistilBERT": ("distilbert-base-uncased", DistilBertTokenizer, DistilBertModel),
    "BERT":      ("bert-base-uncased",        BertTokenizer,       BertModel),
    "RoBERTa":   ("roberta-base",             RobertaTokenizer,    RobertaModel),
    # OPT 家族用 Auto 模型，注意显存
    "OPT-125M":  ("facebook/opt-125m",        AutoTokenizer,       AutoModel),
    "OPT-350M":  ("facebook/opt-350m",        AutoTokenizer,       AutoModel),
    "OPT-1.3B":  ("facebook/opt-1.3b",        AutoTokenizer,       AutoModel),
}

# ========== 简单的 CBM 头（与你代码一致的接口：X->C->Y） ==========
class XtoCtoY(nn.Module):
    def __init__(self, hidden, n_attr, n_cls_attr, n_label):
        super().__init__()
        self.c_heads = nn.ModuleList([nn.Linear(hidden, n_cls_attr) for _ in range(n_attr)])
        self.y_head  = nn.Linear(hidden, n_label)
    def forward(self, z):
        ylogit = self.y_head(z)
        clogits = [head(z) for head in self.c_heads]
        return ylogit, clogits  # 与你脚本相同语义：第一个是任务，后面是概念

# ========== 数据加载 ==========
def load_cebab():
    ds = load_dataset("CEBaB/CEBaB")
    def _map(x):
        # 任务：5类（1..5）-> 0..4
        y = x["review_majority"] - 1
        txt = x["description"]
        map3 = {"Negative":0, "Positive":1, "unknown":2, "":2, "no majority":2}
        c = [map3[x[k]] for k in ["food_aspect_majority","ambiance_aspect_majority","service_aspect_majority","noise_aspect_majority"]]
        return {"text":txt, "y":y, "c":c}
    tr = ds["train_exclusive"].map(_map)
    va = ds["validation"].map(_map)
    te = ds["test"].map(_map)
    return tr, va, te, 4, 5  # n_attr, n_label

def load_imdbc():
    import pandas as pd
    def _read(p):
        df = pd.read_csv(p)
        out = []
        for _,r in df.iterrows():
            out.append({
                "text": r["text"],
                "y":   int(r["label"]),
                "c":   [int(r["acting"]), int(r["storyline"]), int(r["emotion"]), int(r["cinematography"])]
            })
        return out
    tr, va, te = _read(IMDBC_TRAIN), _read(IMDBC_DEV), _read(IMDBC_TEST)
    return tr, va, te, 4, 2

# ========== 简单 DataLoader ==========
def make_loader(split, tok, maxlen, batch=BATCH, shuffle=False):
    enc = tok([s["text"] for s in split], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
    y   = torch.tensor([s["y"] for s in split], dtype=torch.long)
    c   = torch.tensor([s["c"] for s in split], dtype=torch.long)  # [N, n_attr] in {0,1,2}
    ds  = torch.utils.data.TensorDataset(enc["input_ids"], enc.get("attention_mask", torch.ones_like(enc["input_ids"])), y, c)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)

# ========== 稀疏掩码（SparseCBM）：概念特定 mask ==========
def build_masks(model, head, loader, n_attr, sparsity=0.8, use_hessian=False, steps=50):
    # 返回每个概念一个 0/1 mask，按权重重要性（可选用 Fisher/Hessian 近似）
    params = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([p.detach().flatten() for p in params])
    score = torch.zeros_like(flat)

    if use_hessian:
        # 近似 Fisher：累计 grad^2
        model.eval(); head.eval()
        n = 0
        for i,(ids,att,y,c) in enumerate(loader):
            if i>=steps: break
            ids,att,y,c = ids.to(DEVICE), att.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
            model.zero_grad(); head.zero_grad()
            z = model(ids, attention_mask=att).last_hidden_state.mean(1)
            ylogit,_ = head(z)
            loss = F.cross_entropy(ylogit, y)
            loss.backward()
            g = torch.cat([p.grad.detach().flatten() for p in params])
            score += g.abs()
            n += 1
        score /= max(1,n)
    else:
        # Magnitude
        score = flat.abs()

    # 每个概念共享基础重要性，再各自采样出 mask（为简单起见，用同一重要性排序裁剪）
    k_keep = int((1.0 - sparsity) * flat.numel())
    topk_idx = torch.topk(score, k_keep).indices
    masks = []
    start = 0
    # 把扁平 mask 按参数形状还原
    for _ in range(n_attr):
        per = []
        idx_set = set(topk_idx.tolist())
        off = 0
        ms = []
        for p in params:
            n = p.numel()
            m = torch.zeros(n, dtype=torch.float32, device=p.device)
            m_indices = list(range(off, off+n))
            keep = [i-off for i in m_indices if i in idx_set]
            if keep:
                m[keep] = 1.0
            ms.append(m.view_as(p))
            off += n
        masks.append(ms)
    return masks  # list of list(tensor same shape as param)

def apply_mask_forward(model, masks):
    # 返回一个包装前向：给定概念 k 时仅用对应 mask
    params = [p for p in model.parameters() if p.requires_grad]
    def f(ids, att, k):
        # 暂存原权重
        backups = [p.data.clone() for p in params]
        with torch.no_grad():
            for p,m in zip(params, masks[k]):
                p.data.mul_(m)
        out = model(ids, attention_mask=att)
        # 还原
        with torch.no_grad():
            for p,b in zip(params, backups):
                p.data.copy_(b)
        return out
    return f

# ========== 训练与评测 ==========
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def train_eval(dataset_name, backbone_name, mode, seeds=(42,43,44)):
    # 数据
    if dataset_name=="CEBaB":
        tr, va, te, n_attr, n_label = load_cebab()
    else:
        tr, va, te, n_attr, n_label = load_imdbc()

    # 模型与分词器
    hname, Tok, Mod = BACKBONES[backbone_name]
    tok = Tok.from_pretrained(hname)
    mdl = Mod.from_pretrained(hname).to(DEVICE)
    # 统一池化维度
    with torch.no_grad():
        dummy = tok("hi", return_tensors="pt")
        zdim = mdl(**{k:v.to(DEVICE) for k,v in dummy.items()}).last_hidden_state.shape[-1]

    head = XtoCtoY(zdim, n_attr, 3, n_label).to(DEVICE)  # 3类概念：neg/pos/unknown
    opt = torch.optim.AdamW(list(mdl.parameters())+list(head.parameters()), lr=LR)

    def pooled(out): return out.last_hidden_state.mean(1)

    # DataLoader
    tl = make_loader(tr, tok, MAXLEN, BATCH, True)
    vl = make_loader(va, tok, MAXLEN, BATCH, False)
    xl = make_loader(te, tok, MAXLEN, BATCH, False)

    # Sparse masks（如 mode=SparseCBM 时准备）
    masks = None
    if mode=="SparseCBM":
        # 先用一轮 warmup 学到可用表示
        set_seed(1)
        mdl.train(); head.train()
        for ep in range(1):
            for ids,att,y,c in tqdm(tl, desc="warm", leave=False):
                ids,att,y,c = ids.to(DEVICE), att.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
                z = pooled(mdl(ids, attention_mask=att))
                ylogit,_ = head(z)
                loss = F.cross_entropy(ylogit, y)
                opt.zero_grad(); loss.backward(); opt.step()
        masks = build_masks(mdl, head, vl, n_attr, sparsity=0.8, use_hessian=False)
        masked_forward = apply_mask_forward(mdl, masks)

    # 3 次种子训练并在 test 汇总
    outs = []
    for sd in seeds:
        set_seed(sd)
        mdl.train(); head.train()
        for ep in range(EPOCHS):
            for ids,att,y,c in tqdm(tl, desc=f"{mode}-{backbone_name}-{dataset_name}-ep{ep+1}", leave=False):
                ids,att,y,c = ids.to(DEVICE), att.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
                if mode=="Standard":
                    z = pooled(mdl(ids, attention_mask=att))
                    ylogit,_ = head(z)
                    loss = F.cross_entropy(ylogit, y)
                elif mode in ("CBM","SparseCBM"):
                    if mode=="SparseCBM":
                        # 训练阶段不强制 mask，保持与 CBM 一致；推理时使用概念 mask
                        out = mdl(ids, attention_mask=att)
                    else:
                        out = mdl(ids, attention_mask=att)
                    z = pooled(out)
                    ylogit, clogits = head(z)
                    x2y = F.cross_entropy(ylogit, y)
                    # 展平概念 logits 与标签
                    clog = torch.cat(clogits, dim=0)     # [N*n_attr, 3]
                    ctrg = c.T.contiguous().view(-1)      # [N*n_attr]
                    x2c = F.cross_entropy(clog, ctrg)
                    loss = x2y + LAMBDA_X2C * x2c
                opt.zero_grad(); loss.backward(); opt.step()

        # ---- Test 评测（按论文口径）----
        mdl.eval(); head.eval()
        task_pred, task_true = [], []
        conc_pred, conc_true = [], []
        with torch.no_grad():
            for ids,att,y,c in xl:
                ids,att,y,c = ids.to(DEVICE), att.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
                if mode=="SparseCBM":
                    # 概念级前向：对每个概念用其专属 mask 取特征再过对应头
                    z_all = []
                    base = mdl(ids, attention_mask=att)  # 为了与 y 头一致
                    z_base = pooled(base)
                    # 概念预测
                    for k in range(n_attr):
                        outk = masked_forward(ids, att, k)
                        zk   = pooled(outk)
                        z_all.append(zk)
                    # 聚合概念 logits（逐概念）
                    clogits = [head.c_heads[k](z_all[k]) for k in range(n_attr)]
                    ylogit  = head.y_head(z_base)
                else:
                    out = mdl(ids, attention_mask=att)
                    z   = pooled(out)
                    ylogit, clogits = head(z)

                # 任务
                task_pred.extend(torch.argmax(ylogit, dim=1).cpu().tolist())
                task_true.extend(y.cpu().tolist())
                # 概念
                c_hat = torch.argmax(torch.cat(clogits, dim=0), dim=1)          # [N*n_attr]
                conc_pred.extend(c_hat.cpu().tolist())
                conc_true.extend(c.T.contiguous().view(-1).cpu().tolist())

        task_acc = accuracy_score(task_true, task_pred)
        task_f1  = f1_score(task_true, task_pred, average="macro")
        conc_acc = accuracy_score(conc_true, conc_pred)
        conc_f1  = f1_score(conc_true, conc_pred, average="macro")
        outs.append((conc_acc, conc_f1, task_acc, task_f1))

    arr = np.array(outs)  # [3,4]
    mean = arr.mean(0)
    return {"Concept Acc":mean[0]*100, "Concept F1":mean[1]*100, "Task Acc":mean[2]*100, "Task F1":mean[3]*100}

# ========== 主流程：跑齐表 2 ==========
def main():
    rows = []
    for dataset in ["CEBaB","IMDB-C"]:
        for backbone in ["DistilBERT","BERT","RoBERTa","OPT-125M","OPT-350M","OPT-1.3B"]:
            # 某些模型在 IMDB-C/CEBaB 显存吃紧时可按需跳过
            for mode in ["Standard","CBM","SparseCBM"]:
                print(f"[RUN] {dataset} | {backbone} | {mode}")
                try:
                    res = train_eval(dataset, backbone, mode)
                    rows.append({
                        "Backbone": backbone,
                        "Dataset": dataset,
                        "Setting": mode,
                        "Concept Acc": f"{res['Concept Acc']:.1f}",
                        "Concept Macro F1": f"{res['Concept F1']:.1f}",
                        "Task Acc": f"{res['Task Acc']:.1f}",
                        "Task Macro F1": f"{res['Task F1']:.1f}",
                    })
                except Exception as e:
                    rows.append({
                        "Backbone": backbone, "Dataset": dataset, "Setting": mode,
                        "Concept Acc":"-", "Concept Macro F1":"-", "Task Acc":"-", "Task Macro F1":"-"
                    })
                    print("  -> Skipped / Error:", e)

    df = pd.DataFrame(rows)
    # 按论文展示：每个数据集分别一个表，行=骨干，列=两区块（Concept / Task），子列=Acc/F1，三行分别 Standard/CBM/SparseCBM
    def format_one(dataset):
        sub = df[df.Dataset==dataset].copy()
        out_rows = []
        for bb in BACKBONES.keys():
            r_std = sub[(sub.Backbone==bb)&(sub.Setting=="Standard")]
            r_cbm = sub[(sub.Backbone==bb)&(sub.Setting=="CBM")]
            r_sp  = sub[(sub.Backbone==bb)&(sub.Setting=="SparseCBM")]
            def pick(r, k): return (r.iloc[0][k] if len(r)>0 else "-")
            out_rows.append({
                "Backbone": bb,
                "Standard Task Acc/F1":  f"{pick(r_std,'Task Acc')} / {pick(r_std,'Task Macro F1')}",
                "CBM Concept Acc/F1":    f"{pick(r_cbm,'Concept Acc')} / {pick(r_cbm,'Concept Macro F1')}",
                "CBM Task Acc/F1":       f"{pick(r_cbm,'Task Acc')} / {pick(r_cbm,'Task Macro F1')}",
                "SparseCBM Concept Acc/F1": f"{pick(r_sp,'Concept Acc')} / {pick(r_sp,'Concept Macro F1')}",
                "SparseCBM Task Acc/F1":    f"{pick(r_sp,'Task Acc')} / {pick(r_sp,'Task Macro F1')}",
            })
        return pd.DataFrame(out_rows)

    print("\n=== Table 2 (CEBaB) ===")
    print(format_one("CEBaB"))
    print("\n=== Table 2 (IMDB-C) ===")
    print(format_one("IMDB-C"))

if __name__=="__main__":
    main()