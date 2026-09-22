# Development

This file documents the various procedures related to the development.

## Set up development environment


1. Install uv. For details, please see the [Installation](https://docs.astral.sh/uv/getting-started/installation/) section of the manual of uv.
2. Create a virtual environment and install dependencies. The project is installed in editable mode together with the `dev` dependency group.
   ```bash
   cd /path/to/cloned/repository
   uv sync --python /path/to/python  # or just `uv sync` to let uv pick an interpreter
   ```
3. Install [poethepoet](https://github.com/nat-n/poethepoet) 0.33 or later, a task runner. It detects the uv project and runs each task via `uv run`.
   ```bash
   uv tool install poethepoet  # or you may want to use pipx
   poe lint
   poe type-check
   poe test
   poe docs
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
2. Bump version and add a tag. You can use `uv version --bump [patch|minor|major]` to update the version key in `pyproject.toml` (and `uv.lock`). Below is the procedure for using `patch`.
    ```bash
    version=$(uv version --short --bump patch)
    # or $version = uv version --short --bump patch in PowerShell
    echo $version
    git commit -am "Bump version"
    git push
    git tag $version
    git push origin $version
    ```
3. Publish the package to PyPI. You'll need an API token of PyPI, which you can pass with `--token <token>` or the `UV_PUBLISH_TOKEN` environment variable.
   ```bash
   uv build --clear
   uv publish
   ```
