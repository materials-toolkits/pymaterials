import torch
import pytest
from http.server import HTTPServer, SimpleHTTPRequestHandler

from materials_toolkit.data import StructureData
from materials_toolkit.data import HDF5Dataset

import os, sys
import pickle
import tarfile
import multiprocessing as mp
import hashlib
import glob
import urllib.request


def generate_data_list(count=128):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    structures = []
    for _ in range(count):
        n = torch.randint(4, 128, tuple()).item()

        pos = torch.rand(n, 3).float()
        z = torch.randint(1, 128, (n,)).long()
        y = torch.randn(1, 7).float()
        periodic = torch.randint(0, 2, tuple()).bool().item()

        n_edges = torch.randint(4, 1024, tuple()).item()
        edge_index = torch.randint(0, n, (2, n_edges)).long()
        if periodic:
            cell = torch.matrix_exp(torch.rand(1, 3, 3)).float()
            target_cell = torch.randint(-3, 4, (n_edges, 3)).long()
        else:
            cell = None
            target_cell = None

        n_triplets = torch.randint(4, 1024, tuple()).item()
        triplet_index = torch.randint(0, n_edges, (2, n_triplets)).long()

        num_quadruplets = torch.randint(4, 1024, tuple()).item()
        quadruplet_index = torch.randint(0, n_edges, (3, num_quadruplets)).long()

        struct = StructureData(
            z=z,
            pos=pos,
            cell=cell,
            y=y,
            edge_index=edge_index,
            target_cell=target_cell,
            triplet_index=triplet_index,
            quadruplet_index=quadruplet_index,
            periodic=periodic,
        )
        structures.append(struct)

    return structures


@pytest.fixture(scope="session")
def dataset_tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("dataset")

    structures = generate_data_list()

    with open(os.path.join(path, "ground-truth.pickle"), "wb") as handle:
        pickle.dump(structures, handle)

    dataset_dir = os.path.join(path, "test-data")
    HDF5Dataset.create_dataset(dataset_dir, structures)

    with tarfile.open(dataset_dir + ".tar.gz", "w:gz") as tar:
        for file in glob.glob(os.path.join(dataset_dir, "*")):
            tar.add(file, arcname=os.path.basename(file))

    return path


def server_worker(path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=path, **kwargs)

    httpd = HTTPServer(("", 8000), Handler)
    httpd.serve_forever()


def get_md5_hash(file: str) -> str:
    md5 = hashlib.md5()
    with open(file, "rb") as f:
        while True:
            buffer = f.read(1024)
            if not buffer:
                break
            md5.update(buffer)

    return md5.hexdigest()


def compate_structures_list(
    dataset,
    ground_truth,
    keys=[
        "pos",
        "z",
        "cell",
        "y",
        "num_atoms",
        "batch_atoms",
        "periodic",
        "edge_index",
        "target_cell",
        "num_edges",
        "batch_edges",
        "triplet_index",
        "num_triplets",
        "batch_triplets",
        "quadruplets_index",
        "num_quadruplets",
        "batch_quadruplets",
    ],
):
    assert len(dataset) == len(ground_truth)

    for loaded, real in zip(dataset, ground_truth):
        for key in keys:
            if key not in real:
                continue
            assert (
                getattr(loaded, key).shape == getattr(real, key).shape
            ), f"Shape associated with key {key} don't match"
            assert (
                (getattr(loaded, key) == getattr(real, key)).all().item()
            ), f"Data associated with key {key} don't match {getattr(loaded, key)} {getattr(real, key)}"


class no_proxy:
    def __enter__(self):
        self.proxy_backup = urllib.request.getproxies()
        proxy = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)

    def __exit__(self, type, value, traceback):
        proxy = urllib.request.ProxyHandler(self.proxy_backup)
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)


class http_test_data:
    def __init__(self, dataset_tmp_path):
        self.dataset_tmp_path = dataset_tmp_path

    def __enter__(self):
        self.server_process = mp.Process(
            target=server_worker, args=(self.dataset_tmp_path,)
        )
        self.server_process.start()

    def __exit__(self, type, value, traceback):
        self.server_process.terminate()


@pytest.mark.timeout(30)
def test_download_process(dataset_tmp_path):
    with no_proxy(), http_test_data(dataset_tmp_path):
        dataset = HDF5Dataset(
            root=os.path.join(dataset_tmp_path, "data/test-data"),
            url="http://127.0.0.1:8000/test-data.tar.gz",
            md5=get_md5_hash(os.path.join(dataset_tmp_path, "test-data.tar.gz")),
        )

    import h5py

    f = h5py.File(os.path.join(dataset_tmp_path, "test-data/data.hdf5"))

    with open(os.path.join(dataset_tmp_path, "ground-truth.pickle"), "rb") as fp:
        ground_truth = pickle.load(fp)

    compate_structures_list(dataset, ground_truth)
