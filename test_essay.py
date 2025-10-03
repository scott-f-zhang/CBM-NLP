from main.pipelines.joint import get_cbm_joint

result = get_cbm_joint(
    model_name='bert-base-uncased',
    dataset='essay',
    variant='manual',
    num_epochs=15,
    max_len=512,
    batch_size=8,
    optimizer_lr=1e-5,
)
print(result)