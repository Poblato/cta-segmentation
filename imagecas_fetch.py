import kagglehub

# Download latest version
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas", path="1-200.change2zip")
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas", path="1-200.z01")
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas", path="1-200.z02")
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas", path="1-200.z03")
path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas", path="1-200.z04")

print("Path to dataset files:", path)