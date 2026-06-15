#!/usr/bin/env bash
find /Users/tama/Documents/projects/bli-task -maxdepth 4 -type f \( -name "*.py" -o -name "*.env*" -o -name "*.md" \) | sort
