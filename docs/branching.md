
# Git Workflow

## Branch Strategy

- `main`: Stable project version
- `dev`: Shared development branch
- `feature/*`: Individual development tasks
- `fix/*`: Bug fixes

## Development Process

1. Create a feature branch from dev.
2. Develop and commit changes regularly.
3. Push the branch to GitHub.
4. Open a Pull Request to dev.
5. Request a review from a teammate.
6. Merge using a merge commit.
7. Merge dev into main through a Pull Request.

## Conventions

- No direct commits to main after initialization.
- Preserve feature branch history.
- Use meaningful commit messages.
- At least 40 commits per team member.
- Each feature branch should have a reviewed PR.

## Commit Examples

- feature add FastAPI endpoint
- bugifix: correct data validation
- test: add unit tests
- docs: update documentation

