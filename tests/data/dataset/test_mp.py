import pytest

from materials_toolkit.data.datasets import MaterialsProject

import os


@pytest.fixture(scope="session")
def mp_tmp_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mp")

    return path


@pytest.mark.timeout(30)
def test_materials_project_dataset(mp_tmp_path):
    dataset = MaterialsProject(mp_tmp_path)

    assert len(dataset) == 133420
