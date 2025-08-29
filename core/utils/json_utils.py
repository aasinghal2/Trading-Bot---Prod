"""
JSON Utilities for Trading System

Provides utilities for safely converting complex Python objects (Pandas DataFrames,
NumPy types, datetime objects) to JSON-serializable formats.
"""

import math
from datetime import datetime, date
from typing import Any, Dict, List, Union

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    HAS_NUMPY = True
except ImportError:
    pd = None
    np = None
    HAS_PANDAS = False
    HAS_NUMPY = False


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert Pandas/Numpy types and replace NaN/inf with JSON-safe values.

    Rules:
    - DataFrame → list[dict] with None for missing/inf values
    - Series → dict with None for missing/inf values
    - float NaN/inf → None
    - numpy types → converted to native Python
    - dict/list → recursive
    
    Args:
        obj: Object to sanitize for JSON serialization
        
    Returns:
        JSON-serializable version of the object
    """
    if not HAS_PANDAS:
        return _sanitize_basic(obj)
    
    # Pandas structures
    if isinstance(obj, pd.DataFrame):
        # Convert row dicts and sanitize cell-by-cell (avoids DataFrame.fillna/where)
        records = obj.replace([np.inf, -np.inf], np.nan).to_dict(orient='records')
        clean_records = []
        for rec in records:
            clean_rec = {}
            for k, v in rec.items():
                # Convert numpy scalars
                if isinstance(v, np.generic):
                    v = v.item()
                # Replace NaN/inf
                if isinstance(v, float) and (not math.isfinite(v) or v != v):
                    clean_rec[k] = None
                else:
                    clean_rec[k] = sanitize_for_json(v)
            clean_records.append(clean_rec)
        return clean_records
        
    if isinstance(obj, pd.Series):
        s = obj.replace([np.inf, -np.inf], np.nan)
        s_dict = s.to_dict()
        clean = {}
        for k, v in s_dict.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and (not math.isfinite(v) or v != v):
                clean[k] = None
            else:
                clean[k] = sanitize_for_json(v)
        return clean

    # Numpy scalars
    if HAS_NUMPY and isinstance(obj, (np.generic,)):
        native = obj.item()
        return sanitize_for_json(native)

    # Datetime-like
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if HAS_PANDAS and isinstance(obj, getattr(pd, 'Timestamp', ())):
        return obj.isoformat()
    # numpy datetime64
    if HAS_NUMPY and isinstance(obj, (np.datetime64,)):
        try:
            return pd.to_datetime(obj).isoformat()
        except Exception:
            return str(obj)

    return _sanitize_basic(obj)


def _sanitize_basic(obj: Any) -> Any:
    """Basic sanitization for primitive types and containers."""
    # Primitive numbers
    if isinstance(obj, float):
        if not math.isfinite(obj) or (obj != obj):  # NaN or inf
            return None
        return obj

    # Containers
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [sanitize_for_json(v) for v in obj]

    # Objects with dict representation
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        try:
            return sanitize_for_json(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, '__dict__'):
        try:
            return sanitize_for_json(obj.__dict__)
        except Exception:
            pass

    # Fallback: return as-is
    return obj


def safe_json_dump(obj: Any, **kwargs) -> str:
    """Safely dump object to JSON string with sanitization.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    import json
    
    sanitized = sanitize_for_json(obj)
    # Ensure no NaN values slip through
    kwargs.setdefault('allow_nan', False)
    return json.dumps(sanitized, **kwargs)