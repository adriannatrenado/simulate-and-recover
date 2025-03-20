#!/bin/bash
echo "Running tests..."
./src/main.sh
if [ -f "results.csv" ]; then
    echo "Test passed: results.csv generated."
else
    echo "Test failed: results.csv not found."
fi
