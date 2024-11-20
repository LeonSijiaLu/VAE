def get_config():
    return {
        "input_size": 28 * 28, # input dimension of each digit
        "batch_size": 128,
        "hidden_size": 512,
        "latent_size": 8,
        "epochs": 100,
        "learning_rate": 0.001,
    }