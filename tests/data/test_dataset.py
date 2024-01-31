import torch
import pytest
from http.server import HTTPServer, SimpleHTTPRequestHandler

from materials_toolkit.data import StructureData, HDF5Dataset
from materials_toolkit.data.loader import StructureLoader
from materials_toolkit.data.collate import separate

import os, sys
import pickle
import tarfile
import multiprocessing as mp
import hashlib
import glob
import urllib.request

from .utils import *
from .data_convex_hull import *


@pytest.fixture(scope="package")
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


@pytest.mark.timeout(30)
@pytest.mark.dependency()
def test_download(dataset_tmp_path):
    with no_proxy(), http_test_data(dataset_tmp_path):
        dataset = HDF5Dataset(
            root=os.path.join(dataset_tmp_path, "data/test-data"),
            url="http://127.0.0.1:8000/test-data.tar.gz",
            md5=get_md5_hash(os.path.join(dataset_tmp_path, "test-data.tar.gz")),
        )

    with open(os.path.join(dataset_tmp_path, "ground-truth.pickle"), "rb") as fp:
        ground_truth = pickle.load(fp)

    compate_structures_list(dataset, ground_truth)


@pytest.mark.dependency(depends=["test_download"])
def test_loader(dataset_tmp_path):
    dataset = HDF5Dataset(os.path.join(dataset_tmp_path, "data/test-data"))

    assert len(dataset) == 128

    loader = StructureLoader(dataset, batch_size=32)
    structs = []
    for batch in loader:
        for struct in separate(batch):
            structs.append(struct)

    with open(os.path.join(dataset_tmp_path, "ground-truth.pickle"), "rb") as fp:
        ground_truth = pickle.load(fp)

    compate_structures_list(structs, ground_truth)


def test_convex_hull(dataset_tmp_path):
    root = os.path.join(dataset_tmp_path, "dataset_convex_hull")
    write_hdf5_dataset(root)

    dataset = HDF5Dataset(root, use_convex_hull=True)

    data = dataset.get(None, keys=["energy_above_hull"])

    calculated_e_above_hull = data.energy_above_hull
    real_e_above_hull = get_e_above_hull()

    assert (real_e_above_hull - calculated_e_above_hull).abs().max() < 5e-3


def test_convex_hull_after_preprocess(dataset_tmp_path):
    root = os.path.join(dataset_tmp_path, "dataset_convex_hull_after_preprocess")
    write_hdf5_dataset(root)

    dataset = HDF5Dataset(root)
    dataset.compute_convex_hulls()

    data = dataset.get(None, keys=["energy_above_hull"])

    calculated_e_above_hull = data.energy_above_hull
    real_e_above_hull = get_e_above_hull()

    assert (real_e_above_hull - calculated_e_above_hull).abs().max() < 5e-3
