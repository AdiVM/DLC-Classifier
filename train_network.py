import deeplabcut

config_path = "/n/scratch/users/a/adm808/Sabatini_Lab/ZoneDetection2-adi-2025-04-23/config.yaml"
deeplabcut.train_network(config_path)
deeplabcut.evaluate_network(config_path, plotting=True)