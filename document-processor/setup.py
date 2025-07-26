#!/usr/bin/env python3
"""
Setup script for Document Processor Suite
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="document-processor-suite",
    version="1.0.0",
    author="Alex",
    author_email="alex@example.com",
    description="A comprehensive toolkit for document manipulation, conversion, and automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alex/document-processor-suite",
    py_modules=["document_processor"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business",
        "Topic :: Text Processing",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "doc-processor=document_processor:main",
        ],
    },
    keywords="document, conversion, pdf, docx, markdown, automation, text processing",
    project_urls={
        "Bug Reports": "https://github.com/alex/document-processor-suite/issues",
        "Source": "https://github.com/alex/document-processor-suite",
    },
)
