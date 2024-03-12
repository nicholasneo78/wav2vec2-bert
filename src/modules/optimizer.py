import torch

def AdamW_constant_LR(
    model,
    lr: float,
    weight_decay: float,
) -> torch.optim:
    
    """
    AdamW with constant learning rate for all the model layers
    """

    return torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)

def AdamW_grouped_LLRD(
    model,
    init_lr: float=1e-5, 
    multiplier: float=1.75, 
    weight_decay: float=0.01,
) -> torch.optim:

    '''
    inspired from https://freedium.cfd/https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e
    '''

    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "layer_norm.bias", "layer_norm.weight"]
    
    # lower layers to the higher layers of the model 
    _  = [
        "masked_spec_embed", 
        "feature_projection",
        "encoder.layers.0",
        "encoder.layers.1",
        "encoder.layers.2",
        "encoder.layers.3",
        "encoder.layers.4",
        "encoder.layers.5",
    ]

    set_2 = [f"encoder.layers.{x}" for x in range(6,12)]
    set_3 = [f"encoder.layers.{x}" for x in range(12,18)]
    set_4 = [f"encoder.layers.{x}" for x in range(18,24)]
    final_layers = ['adapter.layers', 'lm_head']

    for i, (name, params) in enumerate(named_parameters):

        weight_decay = 0.0 if any(p in name for p in no_decay) else weight_decay

        # for first set
        lr = init_lr
        # for set 2
        lr = init_lr * multiplier if any(p in name for p in set_2) else lr
        # for set 3
        lr = init_lr * multiplier * 2 if any(p in name for p in set_3) else lr
        # for set 4 
        lr = init_lr * multiplier * 3 if any(p in name for p in set_4) else lr
        # final layers
        lr = init_lr * multiplier * 4 if any(p in name for p in final_layers) else lr

        opt_parameters.append(
            {
                "params": params,
                "weight_decay": weight_decay,
                "lr": lr
            }
        )

    return torch.optim.AdamW(opt_parameters, lr=init_lr)