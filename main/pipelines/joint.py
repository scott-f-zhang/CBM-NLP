import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from ..config.defaults import make_run_config
from ..models.loaders import load_model_and_tokenizer
from ..data.cebab import load_cebab_splits, CEBaBDataset
from ..training.loops_joint import train_epoch_joint, eval_epoch_joint, test_epoch_joint

from run_cebab.cbm_models import ModelXtoCtoY_function


def get_cbm_joint(
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
        default_data_type='aug_cebab',
    )
    cfg.mode = 'joint' if cfg.mode is None else cfg.mode

    lambda_XtoC = 0.5
    is_aux_logits = False
    num_labels = 5
    num_each_concept_classes = 3

    model, tokenizer, hidden_size = load_model_and_tokenizer(cfg.model_name, fasttext_path=fasttext_path)

    splits = load_cebab_splits(cfg.data_type)
    train_ds = CEBaBDataset(splits["train"], tokenizer, cfg.max_len, cfg.data_type)
    val_ds = CEBaBDataset(splits["val"], tokenizer, cfg.max_len, cfg.data_type)
    test_ds = CEBaBDataset(splits["test"], tokenizer, cfg.max_len, cfg.data_type)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    num_concept_labels = 10 if cfg.data_type != 'pure_cebab' else 4

    if cfg.model_name == 'lstm':
        head = ModelXtoCtoY_function(
            concept_classes=num_each_concept_classes, label_classes=num_labels, n_attributes=num_concept_labels,
            bottleneck=True, expand_dim=0, n_class_attr=num_each_concept_classes,
            use_relu=False, use_sigmoid=False, Lstm=True, aux_logits=is_aux_logits,
        )
    else:
        head = ModelXtoCtoY_function(
            concept_classes=num_each_concept_classes, label_classes=num_labels, n_attributes=num_concept_labels,
            bottleneck=True, expand_dim=0, n_class_attr=num_each_concept_classes,
            use_relu=False, use_sigmoid=False, aux_logits=is_aux_logits,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=(cfg.optimizer_lr if cfg.optimizer_lr is not None else 1e-5))
    if cfg.model_name == 'lstm':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.num_epochs):
        train_epoch_joint(
            model,
            head,
            train_loader,
            device,
            {"optimizer": optimizer, "criterion": criterion},
            lambda_XtoC,
            cfg.model_name == 'lstm',
        )
        metrics = eval_epoch_joint(model, head, val_loader, device, cfg.model_name == 'lstm')
        print(
            f"Epoch {epoch + 1}: Val concept Acc = {metrics['concept_acc']*100} "
            f"Val concept Macro F1 = {metrics['concept_macro_f1']*100}"
        )
        print(f"Epoch {epoch + 1}: Val Acc = {metrics['val_acc']*100} Val Macro F1 = {metrics['val_macro_f1']*100}")
        if metrics['val_acc'] > best_acc:
            best_acc = metrics['val_acc']
            torch.save(model, f"./{cfg.model_name}_joint.pth")
            torch.save(head, f"./{cfg.model_name}_ModelXtoCtoY_layer_joint.pth")

    model = torch.load(f"./{cfg.model_name}_joint.pth", weights_only=False)
    head = torch.load(f"./{cfg.model_name}_ModelXtoCtoY_layer_joint.pth", weights_only=False)
    model.to(device)
    head.to(device)

    test_acc, test_macro_f1 = test_epoch_joint(model, head, test_loader, device, cfg.model_name == 'lstm')
    print(f"Epoch 1: Test Acc = {test_acc*100} Test Macro F1 = {test_macro_f1*100}")
    return [(test_acc, test_macro_f1)]
