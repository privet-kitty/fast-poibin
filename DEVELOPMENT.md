# Development

This file documents the various procedures related to the development.

## Set up development environment


1. Install Poetry. For details, please see the [Installation](https://python-poetry.org/docs/#installation) section of the manual of Poetry.
2. Install [poethepoet](https://github.com/nat-n/poethepoet), a task-runner for Poetry.
   ```bash
   pip install poethepoet  # or you may want to use pipx
   ```
3. Create a virtual environment and install dependencies.
   ```bash
   cd /path/to/cloned/repository
   poetry env use /path/to/python  # if you need
   poetry install
   ```
4. Run VSCode. You will need to select the Python interpreter after start-up.
   ```bash
   code .
   ```



## Release


So far I haven't automated the release process in CI. On the other hand, the documentation is automatically deployed to GitHub Pages on push to `main` branch. So you need to be careful to avoid inconsistencies with the latest PyPI version.

Below are the procedures to release a new version to PyPI.


1. Check out the latest `main` branch.
    ```bash
    git checkout main
    git fetch
    git reset --hard origin/main
    ```
2. Bump version and add a tag. You can use `poetry version [patch|minor|major]` to update the version key in `pyproject.toml`. Below is the procedure for using `patch`.
    ```bash
    version=$(poetry version --short patch) 
    # or $version = poetry version --short patch in PowerShell
    echo $version
    git commit -am "Bump version"
    git push
    git tag $version
    git push origin $version
    ```
3. Publish the package to PyPI.
   ```bash
   poetry publish --build
   ```
