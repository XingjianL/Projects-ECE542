# overfit, 74.6% val acc @ 49 epoch, no noise
_config_1 = {
    "val_split" : 1/10,
    "lr" : 0.002,
    "hidden" : 64,
    "activation" : "relu",
    "back_dr" : 0.2,
    "back_layer" : 2,
    "back_units" : 128,
    "interval" : 80,
    "batch_freq" : 1,
    "epochs" : 50
}
# with noise N(0,1), 46.6k params
_config_2 = {
    "val_split" : 1/10,
    "lr" : 0.002,
    "hidden" : 128,
    "activation" : "silu",
    "back_dr" : 0.5,
    "back_layer" : 2,
    "back_units" : 64,
    "interval" : 80,
    "batch_freq" : 1,
    "epochs" : 50
}

_config_3 = {
    "val_split" : 1/10,
    "lr" : 0.002,
    "hidden" : 64,
    "activation" : "relu",
    "back_dr" : 0.2,
    "back_layer" : 2,
    "back_units" : 128,
    "interval" : 80,
    "batch_freq" : 1,
    "epochs" : 50
}