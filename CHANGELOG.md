# Changelog

## [0.1.2] - 2025-10-26
### Changed
- Normalize embeddings to unit length before storing and querying.
- Convert Chroma distances to bounded similarity using `1/(1+distance)` to avoid negative similarity values and improve comparability.
- Add runtime checks to validate embedding dimensionality when adding documents to the vector store.

### Notes
- After upgrading, delete your Chroma persist directory and reindex documents to ensure all stored vectors are normalized.
- Suggested reindex command (PowerShell):

```powershell
Remove-Item -Recurse -Force "data\vector_store"
```

Then re-run your ingestion script to rebuild the collection with normalized embeddings.

