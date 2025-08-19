# tests/conftest.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import textwrap
from app.chunking_structure import chunk_by_heading, smart_chunk, chunk_document

DOC = textwrap.dedent("""
6 General Provisions
This is a preface paragraph.

6.1 Safety
This section has multiple paragraphs.

A second paragraph that continues the topic.

6.1.1 PPE
Always wear PPE.

6.1.1.1 Extra detail that should fold into 6.1.1
Still about PPE.

6.2 Conduct
Short.
""").strip()

def test_heading_fold_and_grouping():
    chunks = chunk_by_heading(DOC)
    hs = [h for h, _ in chunks if h]
    assert "6 General Provisions" in hs
    assert "6.1 Safety" in hs
    # Depth >3 folds into 6.1.1
    assert any(h.startswith("6.1.1 ") for h, _ in chunks)

def test_paragraph_never_cut():
    p1 = ("Para1 word " * 120).strip()
    p2 = ("Para2 word " * 120).strip()
    body = p1 + "\n\n" + p2

    parts = smart_chunk(body, max_words=150, min_words=50)

    # Two chunks. Each is exactly one original paragraph.
    assert len(parts) == 2
    assert parts[0] == p1
    assert parts[1] == p2

    # Sanity: no cross-paragraph merges.
    assert "Para1 word Para2" not in parts[0]
    assert "Para1 word Para2" not in parts[1]


def test_chunk_document_combines_policies():
    out = chunk_document(DOC, max_words=60, min_words=20)
    assert out  # non-empty
    # All outputs are (heading, content) and not empty
    assert all(isinstance(h, str) and isinstance(c, str) and c for h, c in out)
