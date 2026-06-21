import kagglehub

# Download latest version
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas", path="1-200.change2zip")

print("Path to dataset files:", path)