"""Download robot model assets — redirects to strands_robots.tools.download_assets.

The download tool is strands_robots/tools/download_assets.py and downloads to
~/.strands_robots/assets/ (user cache) instead of the bundled package dir.
"""

from strands_robots.tools.download_assets import (
    download_assets,
    download_robots,
    get_user_assets_dir,
    main,
)

from strands_robots.tools.download_assets import _needs_download  # noqa: F401

__all__ = ["download_assets", "download_robots", "get_user_assets_dir", "main"]

if __name__ == "__main__":
    main()
