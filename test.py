from materials_toolkit.data.datasets import MaterialsProject

dataset = MaterialsProject("data/mp")

print(dataset[0].to_namedtuple())
