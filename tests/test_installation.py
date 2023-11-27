def test_version():
    import toml
    import materials_toolkit

    data = toml.load("pyproject.toml")

    assert materials_toolkit.__version__ == data["tool"]["poetry"]["version"]
