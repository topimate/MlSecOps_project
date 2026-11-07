
# Git Workflow

## Branch Strategy

- `main`: Stable project version
- `dev`: Shared development branch
- `feature/*`: Individual development tasks
- `fix/*`: Bug fixes

## Development Process – Step by Step

### 1. Switch to the latest dev branch

Before starting a new task, open the VS Code terminal and run:

```bash
git checkout dev
git pull origin dev
```

### 2. Create a new feature branch

Always create a separate branch for your task.

```bash
git checkout -b feature/branch-name
```

Example:

```bash
git checkout -b feature/fastapi-setup
```

### 3. Make changes and commit

After modifying or creating files in VS Code:

```bash
git status
git add .
git commit -m "feat: add FastAPI setup"
```

- `git status` – Shows changed files.
- `git add .` – Stages all changes.
- `git add filename` – Stages only a specific file.
- `git commit -m "message"` – Saves changes locally.

### 4. Push changes to GitHub

For the first push of your feature branch:

```bash
git push -u origin feature/fastapi-setup
```

For later commits on the same branch:

```bash
git add .
git commit -m "feat: add predict endpoint"
git push
```

### 5. Create a Pull Request (PR)

After the first push, GitHub usually displays a link in the terminal:

`Create a pull request for 'feature/fastapi-setup' on GitHub by visiting: ...`

1. **Ctrl + Click** the link to open GitHub.
2. Set **base: dev** and **compare: feature/fastapi-setup**.
3. Add a short description and click **Create Pull Request**.
4. Select another team member under **Reviewers**.
5. Wait for approval, then click **Merge pull request** (use a merge commit, not Squash).

If no link appears, open GitHub → **Pull requests → New pull request**.

### 6. Update dev and start the next task

After merging the PR:

```bash
git checkout dev
git pull origin dev
```

For your next task, create a **new feature branch** from the updated `dev` and repeat the process.

**Important:** Never commit directly to `main` or `dev`. Always use feature branches and Pull Requests.

## Conventions

- No direct commits to main after initialization.
- Preserve feature branch history.
- Use meaningful commit messages.
- At least 50 commits per team member.
- Each feature branch should have a reviewed PR.

## Commit Examples

- feature add FastAPI endpoint
- bugifix: correct data validation
- test: add unit tests
- docs: update documentation
