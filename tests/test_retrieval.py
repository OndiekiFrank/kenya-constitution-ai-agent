import os
import pytest
from src.retrieval import retrieve

def test_constitution_file_exists():
    """Check that constitution.json exists."""
    data_path = os.path.join("src", "constitution.json")
    assert os.path.exists(data_path), "constitution.json file is missing"

def test_retrieve_basic_query():
    """Check that retrieve returns results for a simple query."""
    results = retrieve("What is the role of the President?", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "President" in " ".join(results).capitalize()
