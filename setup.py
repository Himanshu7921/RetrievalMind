from setuptools import setup, find_packages

setup(
    name="RetrievalMind",
    version="0.1.3",
    author="Himanshu Singh",
    author_email="himanshu@example.com",
    description="A custom Retrieval-Augmented Generation (RAG) framework for AI Agent applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Himanshu7921/RetrievalMind",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "langchain",
        "chromadb",
        "sentence-transformers",
        "PyMuPDF"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)