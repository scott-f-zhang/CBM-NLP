from typing import Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from ..config.defaults import make_run_config
from ..models.loaders import load_model_and_tokenizer
from ..data.cebab import CEBaBDataset
from ..data.imdb import IMDBDataset
from ..training.mixup import mixup_hidden_concept, MixupLoss

from run_cebab.cbm_models import ModelXtoCtoY_function


def get_cbm_LLM_mix_joint(
    mode=None,
    max_len=None,
    batch_size=None,
    model_name=None,
    num_epochs=None,
    optimizer_lr=None,
    dataset: Optional[str] = None,
    variant: Optional[str] = None,
    fasttext_path: Optional[str] = None,
):
    cfg = make_run_config(
        mode=mode, max_len=max_len, batch_size=batch_size, model_name=model_name,
        num_epochs=num_epochs, optimizer_lr=optimizer_lr,
        dataset=dataset, variant=variant,
        default_dataset='cebab', default_variant='aug_both',
    )
    cfg.mode = 'joint' if cfg.mode is None else cfg.mode

    if cfg.model_name not in ['bert-base-uncased', 'roberta-base', 'gpt2']:
        return [(0, 0)]

    # Only run for D^ style variants on CEBaB; skip pure D
    if cfg.dataset == 'cebab' and cfg.variant in ['pure']:
        print("[CBE-PLMs-CM] Skipping pure (D) for CM as per paper design")
        return {'task': [], 'concept': []}

    num_labels = 5
    num_each_concept_classes = 3

    model, tokenizer, hidden_size = load_model_and_tokenizer(cfg.model_name)

    if cfg.dataset == 'imdb':
        train_ds = IMDBDataset("train", tokenizer, cfg.max_len, variant=cfg.variant)
        test_ds = IMDBDataset("test", tokenizer, cfg.max_len, variant=cfg.variant)
        num_labels = 2
        num_concept_labels = 8 if getattr(train_ds, "extra", None) is not None else 4
    else:
        train_ds = CEBaBDataset("train", tokenizer, cfg.max_len, variant=cfg.variant)
        test_ds = CEBaBDataset("test", tokenizer, cfg.max_len, variant=cfg.variant)
        num_labels = 5
        num_concept_labels = 10 if getattr(train_ds, "extra", None) is not None else 4

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # num_concept_labels already set above per dataset

    head = ModelXtoCtoY_function(
        concept_classes=num_each_concept_classes, label_classes=num_labels, n_attributes=num_concept_labels,
        bottleneck=True, expand_dim=0, n_class_attr=num_each_concept_classes,
        use_relu=False, use_sigmoid=False, aux_logits=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=(cfg.optimizer_lr if cfg.optimizer_lr is not None else 1e-5))
    loss_fn = MixupLoss()

    scores = []
    concept_scores = []
    for epoch in range(cfg.num_epochs):
        head.train()
        model.train()
        for batch in tqdm(train_loader, desc="Training", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels = batch["concept_labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(1)

            all_h, c_a, c_b, y_a, y_b, lam = mixup_hidden_concept(pooled, concept_labels, label, alpha=0.4, device=device)
            c_a = torch.t(c_a).contiguous().view(-1)
            c_b = torch.t(c_b).contiguous().view(-1)

            outputs2 = head(all_h)
            XtoC_output = outputs2[1:]
            XtoY_output = outputs2[0:1]

            XtoC_logits = torch.nn.Sigmoid()(torch.cat(XtoC_output, dim=0))
            XtoC_loss = loss_fn(XtoC_logits, c_a, c_b, lam)
            XtoY_loss = loss_fn(XtoY_output[0], y_a, y_b, lam)
            loss = 0.5 * XtoC_loss + XtoY_loss
            loss.backward()
            optimizer.step()

        model.eval()
        head.eval()
        predict_labels = np.array([])
        true_labels = np.array([])
        concept_predict_labels = np.array([])
        concept_true_labels = np.array([])
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test", unit="batch"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                concept_labels = batch["concept_labels"].to(device)
                concept_labels = torch.t(concept_labels).contiguous().view(-1)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state.mean(1)
                outputs2 = head(pooled)
                XtoY_output = outputs2[0:1]
                predictions = torch.argmax(XtoY_output[0], axis=1)
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())
                # concept predictions
                XtoC_output = outputs2[1:]
                XtoC_logits = torch.cat(XtoC_output, dim=0)
                concept_predictions = torch.argmax(XtoC_logits, axis=1)
                concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
                concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
        acc = float((predict_labels == true_labels).sum()) / float(len(true_labels)) if len(true_labels) else 0.0
        if len(true_labels):
            labels_unique = np.unique(true_labels)
            macro_f1_scores = []
            for lbl in labels_unique:
                macro_f1_scores.append(f1_score(true_labels == lbl, predict_labels == lbl, average='macro'))
            macro_f1 = float(np.mean(macro_f1_scores))
        else:
            macro_f1 = 0.0
        print(f"Epoch {epoch + 1}: Test Acc = {acc*100} Test Macro F1 = {macro_f1*100}")
        scores.append((acc, macro_f1))
        # concept metrics
        c_acc = float((concept_predict_labels == concept_true_labels).sum()) / float(len(concept_true_labels)) if len(concept_true_labels) else 0.0
        if len(concept_true_labels):
            c_unique = np.unique(concept_true_labels)
            c_macro = []
            for c in c_unique:
                c_macro.append(f1_score(concept_true_labels == c, concept_predict_labels == c, average='macro'))
            c_f1 = float(np.mean(c_macro))
        else:
            c_f1 = 0.0
        concept_scores.append((c_acc, c_f1))

    return {'task': scores, 'concept': concept_scores}
