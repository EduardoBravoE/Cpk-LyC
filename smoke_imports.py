# smoke_imports.py
# Small smoke test to verify package imports for deployment environments (Streamlit Cloud).
# Run locally or on the server to check that `UTILS` package is discoverable.

import sys
from pathlib import Path

print(f"Python executable: {sys.executable}")
print(f"CWD: {Path.cwd()}")
print(f"sys.path (first entries): {sys.path[:5]}")

try:
    import UTILS.common as common
    import UTILS.insights as insights
    import UTILS.lineas_dashboard as ldb
    import UTILS.coples_dashboard as cdb
    print("OK: Imported UTILS.common, UTILS.insights, UTILS.lineas_dashboard, UTILS.coples_dashboard")
    # Show minor sanity info
    print('UTILS.common.get_paths ->', hasattr(common, 'get_paths'))
    print('UTILS.insights functions ->', [name for name in dir(insights) if name.startswith('compute_')][:5])
except Exception as e:
    print('IMPORT ERROR:', e)
    raise
