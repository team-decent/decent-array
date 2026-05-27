Developer Guide
===============
Want to contribute to decent-array? That's great! This guide contains useful information
about development tools, processes, and rules.



Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~
* `Python 3.13+ <https://www.python.org/downloads/>`_
* `tox <https://tox.wiki/en/stable/installation.html>`_

Installation for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A compatible C compiler is required to build the mypyc extensions. On Windows, install 
`Build Tools for Visual Studio 2022 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>`_ or
`C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.
and select the "C++ build tools" workload. On Mac, install Xcode Command Line Tools by running :code:`xcode-select --install` in the terminal.
On Linux, install the appropriate build tools for your distribution, see the `Mypyc documentation <https://mypyc.readthedocs.io/en/latest/getting_started.html>`_ for details.

.. code-block::

   git clone https://github.com/team-decent/decent-array.git
   cd decent-array
   tox -e dev                           # create dev env (admin privileges may be needed)
   source .tox/dev/bin/activate         # activate dev env on Mac/Linux
   .\.tox\dev\Scripts\activate          # activate dev env on Windows

Optionally install development dependencies with proper gpu support, e.g. for PyTorch and TensorFlow:

.. code-block::

   tox -e dev-gpu

It is not recommended to use the development environments for regular usage of decent-array, as they
contain additional packages that are not needed for that purpose. This may cause performance degradation
due to multiple packages competing for resources (e.g. GPU resources).

Tooling
-------
To make sure all GitHub status checks pass, simply run :code:`tox`. You can also run individual checks:

.. code-block::

    tox -e mypy       # find typing issues
    tox -e ruff       # find formatting and style issues
    tox -e pytest     # run tests
    tox -e sphinx     # rebuild documentation
    tox -m lint       # run all linters (ruff + mypy)

Note: Running :code:`tox` commands can take several minutes and may require admin privileges.
If you have mypy addon installed in your IDE, you can use it to get instant feedback on typing issues while coding.
If mypy fails with ``KeyError: 'setter_type'``, delete the ``.mypy_cache`` folder in the project root.

Tools can also be used directly (instead of via tox) after activating the dev environment. Useful examples include:

.. code-block::

    ruff check decent_array --fix                           # find and fix style issues
    ruff format decent_array                                # format code
    mypy decent_array --strict                              # find typing issues
    pytest tests                                            # run tests
    sphinx-build -W -E -b html docs/source docs/build/html  # rebuild html doc files

To verify that doc changes look good, use an html previewer such as
`Live Preview <https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server>`_.
If you are running :code:`pytest tests` while using ``WSL`` on Windows and it starts to randomly fail (or if its really slow), restart your ``WSL`` instance.



Compiled hot path (mypyc)
-------------------------
The :class:`~decent_array.Array` wrapper and the entire
:mod:`~decent_array.interoperability` package are compiled to C extensions with
`mypyc <https://mypyc.readthedocs.io/>`_ at wheel-build time, so end users get the
speed-up automatically when installing from a wheel. During development you can
manage the compiled artifacts with two tox environments:

.. code-block::

    tox -e mypyc          # compile in-place (drops .so files alongside the .py source)
    tox -e clean-mypyc    # remove compiled .so files so Python loads the .py source again

When both ``module.py`` and ``module.cpython-*.so`` exist in the same directory, Python
imports the ``.so``. This means edits to a compiled module will not take effect until
either ``tox -e mypyc`` is rerun or ``tox -e clean-mypyc`` is invoked. A typical
edit-test-edit cycle therefore looks like:

1. Run :code:`tox -e clean-mypyc` once at the start of the session.
2. Edit, run, repeat (Python loads the ``.py`` source directly — no rebuild needed).
3. When measuring performance, run :code:`tox -e mypyc` to recompile and re-bench.

Compilation takes ~25 s from a clean state. The build emits a hash-named shared-runtime
file (``<hash>__mypyc.cpython-*.so``) at the project root that holds helpers used by
every compiled module; it must sit on ``sys.path`` because the compiled modules import
it by bare name. The file is gitignored.



Performance tools
-----------------
The :code:`benchmarks/` directory contains scripts to measure wrapper overhead and
performance of the interoperability layer.


CUTE Design Principles
----------------------
CUTE is a set of principles that serve as guidelines for code design. They are meant to help keep the
codebase simple and the development fast. To mitigate any conflict, the principles are ordered from most to least
important:

1.  **Correctness**: working code is the top priority.
2.  **Understandability**: others should easily understand your code, avoid bloat, unnecessary indirection, and fancy
    abstractions.
3.  **Testability**: code allows for short and clear tests.
4.  **Extendability**: code allows for future extension, but avoid premature generalization and keep YAGNI and KISS in
    mind as trying to predict tomorrow's requirements can cause more problems than it solves.



Pull Requests
-------------
To give other contributors an opportunity to review and to run GitHub status checks, we use pull requests instead of
merging directly to main. The process is detailed below:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Update documentation as needed.
5. Run :code:`tox` to ensure that all checks pass.
6. Submit a pull request.
7. Doc changes? Click the readthedocs link found in the status checks to verify that everything looks good.



Commit Messages
---------------
To keep the git history easy to follow, encourage well-scoped PRs, and facilitate changelog writing and versioning, we
follow certain rules for commit messages when merging pull requests into main. Each message uses this template:

.. code-block:: bash
    :caption: Template

    <type>(<scope>): <subject> (#<pr-id>)

    <description>

    closes #<issue-id>

.. code-block:: bash
   :caption: Example

    perf(costs): Cache m_cvx and m_smooth (#105)

    Cache the properties m_cvx and m_smooth where applicable. This led to a
    75% speed up when running ADMM on a logistic regression problem.

    closes #101

Notes:
    - See table below for types.
    - Scope can be a subpackage, module or build tool, e.g. metrics, costs, or sphinx.
    - Max 72 characters per line.
    - Capitalize but do not punctuate subject.
    - Start subject and description with a verb.
    - Use imperative mood in subject and description.
    - Description explains what changes and why it changes.
    - If the PR has a related issue but doesn't close it, skip the "closes"-keyword and simply reference the issue.

.. list-table::
    :widths: 15 40
    :header-rows: 1
    
    * - Type
      - Description
    * - feat
      - New functionality
    * - perf
      - Performance improvement
    * - ref
      - Refactor
    * - enh
      - Small improvement that doesn't qualify as feat, perf, or ref, e.g. improved variable naming, additional logging,
        or prettier plots
    * - fix
      - Bug fix
    * - test
      - Change to tests
    * - docs
      - Update to readme, comments, docstrings, rst files, or sphinx config
    * - ci
      - CI related change, e.g. modifying GitHub checks or tox environments
    * - meta
      - Update to metadata, e.g. project description, version, or .gitignore
    * - license
      - License update

Inspired by `Sentry <https://develop.sentry.dev/engineering-practices/commit-messages/>`_.
 


Releases
--------
1. Update the version in pyproject.toml using `Semantic Versioning <https://semver.org/>`_.
2. Merge the change into main with commit message :code:`meta: Bump version to <x>.<y>.<z> (#<pr-id>)`.
3. Create a new release on GitHub.
4. Publish to PyPI using :code:`hatch clean && hatch build && hatch publish`.
