#!/bin/bash
# Helper script to create a pull request with prefilled content

# PR Title
TITLE="Add Comprehensive Pre-Commit Hook Integration"

# Branch name
BRANCH="claude/investigate-repo-functions-0154aUYeTuBQB5cr6mkwnKZN"

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found. Creating PR..."
    gh pr create \
        --title "$TITLE" \
        --body-file PULL_REQUEST.md \
        --base main \
        --head "$BRANCH"

    if [ $? -eq 0 ]; then
        echo "✅ Pull request created successfully!"
    else
        echo "❌ Failed to create PR. You may need to authenticate with 'gh auth login'"
    fi
else
    echo "⚠️  GitHub CLI (gh) not found."
    echo ""
    echo "Option 1: Install GitHub CLI"
    echo "  macOS: brew install gh"
    echo "  Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    echo ""
    echo "Option 2: Create PR manually"
    echo "  1. Visit: https://github.com/nhangen/model-checkpoint-engine/pull/new/$BRANCH"
    echo "  2. Copy content from PULL_REQUEST.md"
    echo "  3. Paste into the PR description"
    echo ""
    echo "Quick copy command:"
    echo "  cat PULL_REQUEST.md"
fi
