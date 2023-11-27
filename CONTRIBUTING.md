
# How to contribute

* Open an issue
* Describe your change and discuss what can be done.
* If your proposal is accepted, a new project is created.
* When your project is finished, open a PR (pull request).

# Coding Standards

* We use [Poetry](https://python-poetry.org) to build our package.
    * The `test` group is used for unit testing.
    * The `docs` group is used to build the documentation.
* Coding style
    * Files use 4 space indentation and utf-8 encoding.
    * Use snake case to format variables and Pascal case for class names.
    * Code must be formatted with [black](https://black.readthedocs.io/en/stable/).
    * All modules must include a `__init__.py` file.
* Documentation
    * You must use docstring to document your code.
    * We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate documentation.
    * You must use [Type Hints](https://peps.python.org/pep-0484/) on all public methods and functions.
    * You must use [docstring](https://peps.python.org/pep-0257/) to document every module, file, class, method and function. Your docstring must describe input variables and returned variables.
    * We use [sphinx.ext.doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) to test code snippets.
* Software Design
    * This package relies heavily on [PyTorch](https://pytorch.org/) and [PyTorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/). New features must be designed to be used with the above package.
    * Use OOP when designing new features, but be pragmatic and don't use too much abstraction.
    * The patterns used in a module must be documented.
    * Use must focus on readability and maintainability before abstraction and performance. Prefer simple design patterns and avoid premature optimisation. Some of your code may be highly optimised and less readable when it's needed, but it must be a small part of the codebase.
    * Don't create technical debt and focus on code quality.
* Unit testing
    * We use [pytest](https://docs.pytest.org/en/7.4.x/) for unit testing.
    * Every module must have unit tests. Major functionality must always be tested.
