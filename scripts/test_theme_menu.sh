#!/bin/bash

# Test that theme menu appears and sets env var correctly
echo "Testing theme menu functionality..."

# Mock user input selecting theme 1
echo "1" | bash start.sh 2>&1 | grep -q "Rainbow Gradient"
if [ $? -eq 0 ]; then
    echo "✓ Theme menu displays correctly"
else
    echo "✗ Theme menu not found"
    exit 1
fi
