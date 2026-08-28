"""Configure the pinned WebShop checkout and full catalog."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from opportunity_forecasting import REPO_ROOT

_REPO_ROOT = REPO_ROOT
def _resolve_webshop_root() -> Path:
    """Return the populated WebShop checkout for this repo layout."""

    candidates = [
        _REPO_ROOT / "WebShop",
        _REPO_ROOT / "third_party" / "WebShop",
    ]
    for candidate in candidates:
        if (candidate / "web_agent_site").is_dir():
            return candidate


    return candidates[0]


_WEBSHOP_ROOT = _resolve_webshop_root()
_WEBSHOP_DATA_DIR = Path(
    os.environ.setdefault("WEBSHOP_DATA_DIR", str(_WEBSHOP_ROOT / "data"))
).resolve()
_WEBSHOP_SEARCH_ENGINE_DIR = Path(
    os.environ.setdefault("WEBSHOP_SEARCH_ENGINE_DIR", str(_WEBSHOP_DATA_DIR.parent / "search_engine"))
).resolve()


def configure_java_runtime() -> None:
    environment_prefix = Path(sys.prefix)
    java_home = environment_prefix / "lib" / "jvm"
    libjvm = java_home / "lib" / "server" / "libjvm.so"
    if not libjvm.is_file():
        raise FileNotFoundError(libjvm)
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["JVM_PATH"] = str(libjvm)
    os.environ["PATH"] = os.pathsep.join(
        (
            str(environment_prefix / "bin"),
            str(java_home / "bin"),
            os.environ.get("PATH", ""),
        )
    )


def ensure_public_webshop_imports() -> Path:
    """
    Make the public WebShop submodule importable and prefer full-data defaults.

    Returns the resolved WebShop data directory.
    """

    configure_java_runtime()
    webshop_root_txt = str(_WEBSHOP_ROOT)
    if webshop_root_txt not in sys.path:
        sys.path.insert(0, webshop_root_txt)

    import web_agent_site.utils as ws_utils

    full_file_path = _WEBSHOP_DATA_DIR / "items_shuffle.json"
    full_attr_path = _WEBSHOP_DATA_DIR / "items_ins_v2.json"
    full_human_attr_path = _WEBSHOP_DATA_DIR / "items_human_ins.json"
    if full_file_path.exists():
        ws_utils.DEFAULT_FILE_PATH = str(full_file_path)
    if full_attr_path.exists():
        ws_utils.DEFAULT_ATTR_PATH = str(full_attr_path)
    if full_human_attr_path.exists():
        ws_utils.HUMAN_ATTR_PATH = str(full_human_attr_path)

    if _WEBSHOP_SEARCH_ENGINE_DIR.exists():
        from web_agent_site.engine import engine as ws_engine

        def init_search_engine(num_products=None):
            if num_products == 100:
                indexes = "indexes_100"
            elif num_products == 1000:
                indexes = "indexes_1k"
            elif num_products == 100000:
                indexes = "indexes_100k"
            elif num_products is None:
                indexes = "indexes"
            else:
                raise NotImplementedError(f"num_products being {num_products} is not supported yet.")
            return ws_engine.LuceneSearcher(str(_WEBSHOP_SEARCH_ENGINE_DIR / indexes))

        ws_engine.init_search_engine = init_search_engine

    return _WEBSHOP_DATA_DIR
