#!/bin/sh

py_lint() {
    isort $1
    yapf -i $1
}

py_lint tests/read_mmid.py
