#!/bin/bash
# CoMarketer Claude Code Setup Script
# Run this from your project root: bash setup_claude_code.sh
#
# This script copies the Claude Code configuration into your existing repo.
# It does NOT modify any of your source code.

set -e

echo "=== CoMarketer Claude Code Setup ==="
echo ""

# Check we're in the right directory
if [ ! -d "agent_server" ]; then
    echo "ERROR: Run this from the project root (where agent_server/ exists)"
    exit 1
fi

# Backup existing CLAUDE.md if present
if [ -f "CLAUDE.md" ]; then
    echo "→ Backing up existing CLAUDE.md to CLAUDE.md.backup"
    cp CLAUDE.md CLAUDE.md.backup
fi

# Backup existing .claude if present
if [ -d ".claude" ]; then
    echo "→ Backing up existing .claude/ to .claude.backup/"
    cp -r .claude .claude.backup
fi

# Create directory structure
echo "→ Creating .claude/ directory structure..."
mkdir -p .claude/skills/{architecture,langgraph-databricks,genie-mcp,domain-marketing,stats-engine,memory-system,evaluation-framework,streaming-sse,implement-component,review-code,deploy-databricks}
mkdir -p .claude/agents
mkdir -p .claude/rules
mkdir -p docs

echo "→ Copying CLAUDE.md..."
echo "→ Copying settings.json..."
echo "→ Copying skills (11 skills)..."
echo "→ Copying subagents (3 agents)..."
echo "→ Copying rules (3 rule files)..."
echo "→ Copying implementation plan..."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Files created:"
echo "  CLAUDE.md                                    — Project manifest (loaded every session)"
echo "  .claude/settings.json                        — Permissions + auto-lint hook"
echo "  .claude/skills/architecture/SKILL.md          — Hierarchical supervisor patterns"
echo "  .claude/skills/langgraph-databricks/SKILL.md  — LangGraph + Databricks API reference"
echo "  .claude/skills/genie-mcp/SKILL.md             — Genie MCP migration guide"
echo "  .claude/skills/domain-marketing/SKILL.md      — Marketing domain (metrics, verticals, formatting)"
echo "  .claude/skills/stats-engine/SKILL.md           — Stats engine implementation guide"
echo "  .claude/skills/memory-system/SKILL.md          — 4-layer memory architecture"
echo "  .claude/skills/evaluation-framework/SKILL.md   — 6-layer evaluation framework"
echo "  .claude/skills/streaming-sse/SKILL.md          — 8-phase SSE streaming contract"
echo "  .claude/skills/implement-component/SKILL.md    — Step-by-step implementation workflow"
echo "  .claude/skills/review-code/SKILL.md            — Code review checklist"
echo "  .claude/skills/deploy-databricks/SKILL.md      — Deployment workflow"
echo "  .claude/agents/researcher.md                   — Read-only codebase explorer"
echo "  .claude/agents/code-reviewer.md                — Senior engineer code reviewer"
echo "  .claude/agents/test-writer.md                  — Test generation agent"
echo "  .claude/rules/python.md                        — Python conventions (loads for .py files)"
echo "  .claude/rules/langgraph.md                     — LangGraph rules (loads for graph/agent files)"
echo "  .claude/rules/prompts.md                       — Prompt writing rules (loads for prompt files)"
echo "  docs/IMPLEMENTATION_PLAN.md                    — Phased development plan with task prompts"
echo ""
echo "Next steps:"
echo "  1. cd $(pwd)"
echo "  2. Copy your REPO_AUDIT.md into docs/ (if not already there)"
echo "  3. Run: claude"
echo "  4. Verify: /help (should show your custom skills)"
echo "  5. Start with Task 0.1 from docs/IMPLEMENTATION_PLAN.md"
echo ""
echo "Tips:"
echo "  - Use Plan Mode (Shift+Tab) before each task"
echo "  - Use /clear between tasks to keep context clean"
echo "  - Monitor context with /context — compact at 50%"
echo "  - Skills load automatically when relevant — no need to invoke manually"
echo "  - Use 'Use the researcher agent to...' for codebase exploration"
echo "  - Use 'Use the code-reviewer agent to...' after implementation"
