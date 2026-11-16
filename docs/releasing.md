# Releasing IMSTE

IMSTE uses the version in `pyproject.toml` as its single source of release
versioning. The PyPI workflow requires each GitHub Release tag to match that
version with a leading `v`.

## Version numbers

The project is pre-1.0, so releases use `0.MINOR.PATCH`:

- Bump the patch number for backward-compatible fixes, such as `0.1.0` to
  `0.1.1`.
- Bump the minor number for backward-compatible features, such as `0.1.0` to
  `0.2.0`.
- Keep the major number at `0` while the public API is still stabilizing. Move
  to `1.0.0` when the API is considered stable. After `1.0.0`, use standard
  semantic versioning: patch for fixes, minor for compatible features, and
  major for breaking changes.

## First successful PyPI release

The first package version to publish on PyPI is `0.2.1`. Before creating its
GitHub Release:

1. In the PyPI account, create a pending Trusted Publisher for project `imste`
   with owner `fabriziocosta`, repository `IteratedMinimumSpanningTree`,
   workflow file `publish.yml`, and GitHub environment `pypi`.
2. Create and publish a GitHub Release tagged `v0.2.1`.

The workflow runs tests, builds and checks both the wheel and source
distribution, verifies the tag against `pyproject.toml`, then publishes the
same build artifacts to PyPI using Trusted Publishing. The first successful
publish creates the PyPI project from the pending publisher.

For later releases, update `project.version` in `pyproject.toml`, push the
change, and publish a GitHub Release with the matching `v<version>` tag. PyPI
does not allow replacing a published version; publish a new patch version to
correct a release.
