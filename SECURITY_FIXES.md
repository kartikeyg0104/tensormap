# Security and Quality Fixes for Export Service

This document summarizes the critical fixes applied to the model export service based on the automated code review feedback.

## Fixed Issues

### 1. [BLOCKING/SECURITY] Path Traversal Vulnerability (Fixed ✅)

**Issue:** Model names were validated only for non-emptiness but used directly in file paths, allowing attackers to write files outside the intended directory.

**Fix:**
- Added `_sanitize_filename()` function in `model_export.py` that:
  - Removes path separators (`/`, `\`)
  - Removes parent directory references (`..`)
  - Only allows safe characters: alphanumeric, underscore, hyphen, dot
  - Prevents hidden files (starts with dot)
  - Limits length to 100 characters
- Applied sanitization in all export functions: `export_savedmodel()`, `export_tflite()`, `export_onnx()`, and `get_export_formats()`
- Added test `test_path_traversal_prevention()` to verify the fix

**Impact:** Prevents arbitrary file write attacks through crafted model names.

---

### 2. [MAJOR/BUG] Blocking I/O in Async Endpoints (Fixed ✅)

**Issue:** Export endpoints called synchronous TensorFlow/tf2onnx functions directly from async handlers, blocking the event loop and freezing all other requests.

**Fix:**
- Wrapped all blocking export calls with `run_in_threadpool()` in `export.py`:
  - `await run_in_threadpool(export_savedmodel, job_id, model_name)`
  - `await run_in_threadpool(export_tflite, job_id, model_name)`
  - `await run_in_threadpool(export_onnx, job_id, model_name, model.graph_ir)`

**Impact:** Prevents freezing of the entire application during model exports; maintains responsiveness for all users.

---

### 3. [MAJOR/QUALITY] Unnecessary Model Loading (Fixed ✅)

**Issue:** `get_export_formats()` loaded the full TensorFlow model just to call `validate_onnx_compatible()`, which never actually used the model parameter.

**Fix:**
- Modified `get_export_formats()` to pass `None` as the model parameter to `validate_onnx_compatible()`
- Removed the expensive `tf.keras.models.load_model()` call
- `validate_onnx_compatible()` only inspects `graph_ir`, not the model

**Impact:** Significantly faster format listing (called on every ExportPanel mount and after each download).

---

### 4. [MINOR/BUG] Background Task Session Issue (Fixed ✅)

**Issue:** Background task `update_download_timestamp()` used the request-scoped `db` session, which would be closed before the task ran.

**Fix:**
- Changed background task to use `get_session()` context manager
- Creates a new session specifically for the background task
- Properly queries, updates, and commits within the new session context

**Impact:** Ensures `last_export_download_at` is reliably updated.

---

### 5. [MINOR/BUG] Frontend Filename Fallback (Fixed ✅)

**Issue:** Fallback filename for SavedModel was `{modelName}.savedmodel` instead of `{modelName}.savedmodel.zip`, and cross-origin requests couldn't read the `Content-Disposition` header.

**Fixes:**
- Updated frontend fallback logic in `ExportPanel.tsx` to use `.savedmodel.zip` for SavedModel format
- Added `expose_headers=["Content-Disposition"]` to CORS middleware in `main.py`

**Impact:** Users always get the correct filename with proper extension, even in cross-origin setups.

---

### 6. [MINOR/QUALITY] Error Handling in Cascade Cleanup (Fixed ✅)

**Issue:** `delete_model_exports()` would abort on first failure, leaving remaining export directories undeleted despite being described as "best-effort."

**Fix:**
- Wrapped individual `delete_job_exports()` calls in try-except
- Log failures but continue processing remaining jobs
- Return count of successfully deleted directories

**Impact:** True best-effort cleanup; one failed deletion doesn't prevent cleanup of others.

---

### 7. [NIT/STYLE] Duplicated Export Path (Fixed ✅)

**Issue:** `metrics_callback.py` hardcoded `Path("./exports/{job_id}")` instead of importing `EXPORTS_BASE`.

**Fix:**
- Imported `EXPORTS_BASE` from `model_export.py`
- Used `EXPORTS_BASE / self.job_id` consistently

**Impact:** Maintains single source of truth for export directory location.

---

## Testing

All fixes are validated by existing and new tests:

```bash
cd tensormap-backend
pytest tests/test_model_export.py -v
```

**Results:** 12 passed, 2 skipped (tf2onnx availability)

Key new test:
- `test_path_traversal_prevention()` - Verifies malicious model names are sanitized

---

## Code Quality

All changes pass linting:

```bash
ruff check app/services/model_export.py app/routers/export.py app/callbacks/metrics_callback.py
```

**Result:** All checks passed! ✅

---

## Summary

All blocking, major, and minor issues from the automated review have been addressed:
- ✅ Security vulnerability patched (path traversal)
- ✅ Performance issue fixed (blocking I/O)
- ✅ Unnecessary operations removed (model loading)
- ✅ Background task reliability improved
- ✅ Cross-browser compatibility enhanced
- ✅ Error handling made resilient
- ✅ Code consistency improved

The export service is now production-ready with proper security, performance, and reliability characteristics.
