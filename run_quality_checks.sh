#!/bin/bash

echo -e "\nRunning Flake8 style guide enforcement..."
flake8 *.py tests/**/*.py
if [ $? -eq 0 ]; then
    echo "✅ Flake8 check passed"
else
    echo "❌ Flake8 check failed"
fi

echo -e "\nRunning Pylint code analysis..."
pylint *.py tests/**/*.py
if [ $? -eq 0 ]; then
    echo "✅ Pylint check passed"
else
    echo "❌ Pylint check failed"
fi
