# Branch Protection

GitHub template repositories copy files, but they do not copy repository settings. Every repository created from this template must configure branch protection manually.

Use this checklist after creating a new repository from the template.

## 1. Run the checks once

Required checks cannot be selected until they have reported at least once.

1. Push the repository to GitHub.
2. Open a pull request, or push a commit to `main`.
3. Wait for the CI workflow to finish.

## 2. Configure `main`

`Settings -> Branches -> Add branch protection rule`

Set the branch name pattern to:

```text
main
```

Enable these settings:

- Require a pull request before merging.
- Require at least 1 approving review.
- Dismiss stale pull request approvals when new commits are pushed.
- Require conversation resolution before merging.
- Require status checks to pass before merging.
- Require branches to be up to date before merging.
- Require linear history.
- Include administrators.
- Block force pushes.
- Block branch deletion.

Select all three required status checks:

```text
format
test
web
```

They may instead appear in the list as `CI / format`, `CI / test`, and `CI / web`. Choose whichever version GitHub shows you. `web` passes until the `web/` folder exists, and starts checking it once it does.

Leave switched off:

- Require code quality results. It needs GitHub code scanning, which this
  repository does not use, so every pull request would wait on an analysis that
  never runs.

## 3. Pull Request Settings

`Settings -> General -> Pull Requests`

- Enable squash merging.
- Disable merge commits.
- Disable rebase merging.
- Enable automatically delete head branches.

## 4. Actions Permissions

`Settings -> Actions -> General`

- Actions permissions: allow actions created by third parties, or add
  `astral-sh/setup-uv@*` to the allow list. `actions/checkout` and
  `actions/setup-node` are GitHub's own, but `astral-sh/setup-uv` is not, so the
  GitHub-only option is not enough and every job fails immediately.
- Workflow permissions: Read repository contents permission. Each workflow asks
  for anything more at the top of its own file.

If an organisation owns this repository, these settings also exist at organisation level, and the stricter of the two applies.

## 5. Add the GPU Machine

`Settings -> Actions -> Runners -> New self-hosted runner`

Follow the instructions on that page. Add the label `gpu`. `self-hosted` is applied automatically, and the workflow asks for both.

Then add `gpu` back to the `all` group in `.github/scripts/parse-command.sh`, where a TODO marks the line. It is left out until the machine exists so that `/test all` cannot wait on a runner that is not there.

Until this is done, do not use `/test gpu`. GitHub allows a job to sit in the queue waiting for a self-hosted runner for 24 hours before giving up, and the hour long timeout on the job does not apply, because that limits how long a job may run rather than how long it may wait. No result comment is posted until the queue time runs out.