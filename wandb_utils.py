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
        "num_transformer_heads": {"values": [8]},
        "num_transformer_layers": {"values": [8]},
        "transformer_use_rms_norm": {"values": [True, False]},
        "use_fp16": {"values": [True, False]},
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
    },
}
