import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from ..config.defaults import make_run_config
from ..models.loaders import load_model_and_tokenizer
from ..data.cebab import load_cebab_splits, CEBaBDataset
from ..training.loops import train_one_epoch, evaluate, test_loop


def get_cbm_standard(
    mode=None,
    max_len=None,
    batch_size=None,
    model_name=None,
    num_epochs=None,
    data_type=None,
    optimizer_lr=None,
    fasttext_path: str | None = None,
):
    cfg = make_run_config(
        mode=mode, max_len=max_len, batch_size=batch_size, model_name=model_name,
        num_epochs=num_epochs, data_type=data_type, optimizer_lr=optimizer_lr,
        default_data_type='pure_cebab',
    )
    cfg.mode = 'standard' if cfg.mode is None else cfg.mode

    model, tokenizer, hidden_size = load_model_and_tokenizer(cfg.model_name, fasttext_path=fasttext_path)

    splits = load_cebab_splits(cfg.data_type)
    train_ds = CEBaBDataset(splits["train"], tokenizer, cfg.max_len, cfg.data_type)
    val_ds = CEBaBDataset(splits["val"], tokenizer, cfg.max_len, cfg.data_type)
    test_ds = CEBaBDataset(splits["test"], tokenizer, cfg.max_len, cfg.data_type)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    num_labels = 5
    num_concept_labels = 10 if cfg.data_type != 'pure_cebab' else 4
    if cfg.model_name == 'lstm':
        head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Linear(hidden_size // 2, num_concept_labels),
            torch.nn.Linear(num_concept_labels, num_labels),
        )
        scheduler_needed = True
    else:
        head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_concept_labels),
            torch.nn.Linear(num_concept_labels, num_labels),
        )
        scheduler_needed = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=(cfg.optimizer_lr if cfg.optimizer_lr is not None else (1e-2 if cfg.model_name == 'lstm' else 1e-5)))
    if scheduler_needed:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.num_epochs):
        train_one_epoch(model, head, train_loader, device, criterion, optimizer, cfg.model_name == 'lstm')
        val_acc, val_f1 = evaluate(model, head, val_loader, device, cfg.model_name == 'lstm')
        print(f"Epoch {epoch + 1}: Val Acc = {val_acc*100} Val Macro F1 = {val_f1*100}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(head, f"./{cfg.model_name}_classifier_standard.pth")
            torch.save(model, f"./{cfg.model_name}_model_standard.pth")

    # test
    model = torch.load(f"./{cfg.model_name}_model_standard.pth", weights_only=False)
    head = torch.load(f"./{cfg.model_name}_classifier_standard.pth", weights_only=False)
    model.to(device)
    head.to(device)

    scores = test_loop(model, head, test_loader, device, cfg.model_name == 'lstm')
    return scores
