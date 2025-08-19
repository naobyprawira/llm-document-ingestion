"""Command line interfaces for document ingestion pipeline.

This package exposes CLI scripts to ingest documents into Qdrant and to
delete them.  These entry points are thin wrappers around the core
functions in :mod:`app.pipelines.ingest_pdf` and :mod:`app.storage.qdrant_store`.
"""
