sweep_config = {
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "optimizer": {
            "values": [
                "adam",
                "adamW",
                "adagrad",
            ]
        },
        "num_transformer_heads": {"values": [2, 4, 8]},
        "num_transformer_layers": {"values": [2, 4, 8]},
        "transformer_use_rms_norm": {"values": [True, False]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-2,
        },
        "batch_size": {"value": 32},
    },
}
