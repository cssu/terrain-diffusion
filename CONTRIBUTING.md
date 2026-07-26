# Contributing

How to check your work, how tests are organised, and how changes get reviewed and merged. See the [README](README.md) first for how to install the project.

## Running local checks

Committing and pushing are never
blocked. Pre-merge checks run on each PR via GitHub actions which may be slow.

You can run the same checks localy using:

```bash
scripts/quality-check.sh
```


Most formatting check failures can be enforced with:

```bash
scripts/format-check.sh --fix
```

To run more specific tests, refer to the following commands.

| Command | What it runs |
| --- | --- |
| `scripts/quality-check.sh` | All three of the below. The same as GitHub. |
| `scripts/format-check.sh` | Formatting and code style. |
| `scripts/format-check.sh --fix` | Corrects the formatting and style problems that can be corrected. |
| `scripts/test-check.sh` | The quick tests. |
| `scripts/test-check.sh slow` | The tests marked `slow`. |
| `scripts/test-check.sh gpu` | The tests marked `gpu`. |
| `scripts/test-check.sh all` | Every test. |
| `scripts/web-check.sh` | The visualizer, once `web/` exists. |


Testing in this project is mostly done with pytest (web test framework is undecided). We use pytest markers to categorize tests into three groups: `quick`, `slow`, and `gpu`.
`slow` and `gpu` tests are explicitly marked using `@pytest.mark.<slow|gpu>`, a test is categorized as `quick` otherwise.
You can use `test-check.sh` to run tests locally. Refer to the table above on how to invoke specific groups.

Anything after the group are treated as pytest args, so
`scripts/test-check.sh python -k sampler` passes the `-k sample` argument to pytest.

You can also run pytest directly, as long as you go through uv so that the
project's environment is used:

```bash
uv run pytest
uv run pytest -k sampler
```

## Adding a package

To add a package, put it in `pyproject.toml`, run `uv sync`, and commit the change to `uv.lock`. Skipping the lockfile means the package works for you and breaks for everyone else.

## Opening a pull request

1. Make a branch. Do not commit to `main` directly.
2. Push your branch and open a pull request into `main`.
3. Fill in the pull request description.
4. Wait for the checks to pass. GitHub will not let the pull request merge until they do.
5. Run appropriate tests using a PR comment (see below)
6. Wait for approval and ping relevant people.
7. Merge.


One thing to expect: pushing a new commit also removes any approval the pull
request already had, so it needs approving again. That is deliberate, so that
what gets approved is what gets merged.

The GitHub settings that make these checks compulsory are not stored in the repository. They are listed in [docs/branch-protection.md](docs/branch-protection.md).

## Testing

Every PR is expected to show some proof of testing for their work. This can be with unit tests or with new pytests added to one of the three groups below.

**Quick tests** have no marker on them. They run on every push and
decide whether a pull request is allowed to merge. Most tests should be in this
group. Write them so they stay fast.

**Slow tests** are marked `@pytest.mark.slow`. They are left out when you push,
so they never hold up ordinary work. Run them on request, as described below.

**GPU tests** are marked `@pytest.mark.gpu`. These will only run on the project's own GPU machine, on request.

A GPU test needs two lines on it, not one:

```python
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_core_model_runs():
    ...
```

The marker is what lets the GPU machine pick the test out and run it. The
`skipif` is what lets the test sit harmlessly in the suite everywhere else: on
a machine with no GPU it is reported as skipped instead of failing. Without
`skipif`, running the whole suite would fail for a reason that is not a real
problem with the code.

## Running extra tests on a pull request

Pushing only runs the quick tests. To run more, leave a comment on the pull
request:

| Comment | What it runs |
| --- | --- |
| `/test all` or `/test` | Every group below except `gpu`, at the same time. |
| `/test python` | The quick tests. |
| `/test slow` | The tests marked `slow`. |
| `/test gpu` | The tests marked `gpu`, on the project's GPU machine. |
| `/test web` | The 3D visualizer. |
| `/test help` | Lists these commands. |

Each comment runs its own job on the latest commit pushed to the branch, re-running a command while a job of the same type is running will cancel the previous run and start a new one on the latest commit.

Only people with access to this repository can use them. A short while after
commenting you should see a 👀 reaction on your comment, and then a reply with
the result.

The reply names the exact commit it tested. That matters: if you push more
commits afterwards, the old result no longer applies to the new code, and a
reviewer is meant to notice this and ask for the tests to be run again.

These commands do not block merging on their own. Only the automatic checks do
that. Whether the extra tests were run is something the reviewer checks, using
the checklist in the pull request template.
