#!/usr/bin/env python3
"""
Hugging Face Repository Downloader

This script downloads a Hugging Face repository (model, dataset, or space) 
to a local folder using the huggingface_hub library.
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download, login
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_repository(repo_id, local_dir, repo_type="model", token=None, revision="main",
                        include_patterns=None, exclude_patterns=None, cache_dir=None, force=False):
    """
    Download a Hugging Face repository to a local directory.
    
    Args:
        repo_id (str): Repository ID (e.g., "microsoft/DialoGPT-medium")
        local_dir (str): Local directory to save the repository
        repo_type (str): Type of repository ("model", "dataset", or "space")
        token (str): Hugging Face token for private repositories
        revision (str): Git revision (branch, tag, or commit hash)
        include_patterns (list): List of file patterns to include
        exclude_patterns (list): List of file patterns to exclude
        cache_dir (str): Directory to use for caching
        force (bool): Force download even if directory exists
    
    Returns:
        str: Path to the downloaded repository
    """
    try:
        local_dir_path = Path(local_dir)
        if not force and local_dir_path.exists() and any(local_dir_path.iterdir()):
            logger.info(f"Directory '{local_dir}' already exists and is not empty. Skipping download. Use --force to override.")
            return str(local_dir_path)

        logger.info(f"Starting download of {repo_type} repository: {repo_id}")
        logger.info(f"Revision: {revision}")
        logger.info(f"Local directory: {local_dir}")

        # Create local directory if it doesn't exist
        local_dir_path.mkdir(parents=True, exist_ok=True)

        # Download the repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
            token=token,
            allow_patterns=include_patterns,
            ignore_patterns=exclude_patterns,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False  # Create actual files, not symlinks
        )

        logger.info(f"Successfully downloaded repository to: {downloaded_path}")
        return downloaded_path

    except RepositoryNotFoundError:
        logger.error(f"Repository '{repo_id}' not found. Please check the repository ID.")
        return None
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            logger.error("Authentication failed. Please provide a valid Hugging Face token for private repositories.")
        else:
            logger.error(f"HTTP error occurred: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None

def download_single_file(repo_id, filename, local_dir, repo_type="model", token=None, revision="main", force=False):
    """
    Download a single file from a Hugging Face repository.
    
    Args:
        repo_id (str): Repository ID
        filename (str): Name of the file to download
        local_dir (str): Local directory to save the file
        repo_type (str): Type of repository ("model", "dataset", or "space")
        token (str): Hugging Face token for private repositories
        revision (str): Git revision (branch, tag, or commit hash)
        force (bool): Force download even if file exists
    
    Returns:
        str: Path to the downloaded file
    """
    try:
        target_file_path = Path(local_dir) / filename
        if not force and target_file_path.exists():
            logger.info(f"File '{target_file_path}' already exists. Skipping download. Use --force to override.")
            return str(target_file_path)

        logger.info(f"Downloading file '{filename}' from {repo_id}")

        # Create local directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Download the file
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

        logger.info(f"Successfully downloaded file to: {downloaded_file}")
        return downloaded_file

    except Exception as e:
        logger.error(f"Failed to download file '{filename}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face repositories to local folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model repository
  python huggingface_downloader.py microsoft/DialoGPT-medium ./models/DialoGPT

  # Download a dataset
  python huggingface_downloader.py squad ./datasets/squad --repo-type dataset

  # Download with authentication token
  python huggingface_downloader.py private/model ./models/private --token your_token_here

  # Download specific files only
  python huggingface_downloader.py microsoft/DialoGPT-medium ./models --include "*.json" "*.txt"

  # Download single file
  python huggingface_downloader.py microsoft/DialoGPT-medium ./models --file config.json
  
  # Force download even if destination exists
  python huggingface_downloader.py microsoft/DialoGPT-medium ./models --force
        """
    )

    parser.add_argument("repo_id", help="Repository ID (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("local_dir", help="Local directory to save the repository")
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"],
                        default="model", help="Type of repository (default: model)")
    parser.add_argument("--token", help="Hugging Face token for private repositories")
    parser.add_argument("--revision", default="main",
                        help="Git revision to download (default: main)")
    parser.add_argument("--include", nargs="*",
                        help="File patterns to include (e.g., '*.json' '*.txt')")
    parser.add_argument("--exclude", nargs="*",
                        help="File patterns to exclude (e.g., '*.bin' '*.safetensors')")
    parser.add_argument("--cache-dir", help="Directory to use for caching")
    parser.add_argument("--file", help="Download a single file instead of entire repository")
    parser.add_argument("--login", action="store_true",
                        help="Login to Hugging Face before downloading")
    parser.add_argument("--force", action="store_true",
                        help="Force download even if the destination already exists and is not empty.")

    args = parser.parse_args()

    # Handle login if requested
    if args.login:
        try:
            login(token=args.token)
            logger.info("Successfully logged in to Hugging Face")
        except Exception as e:
            logger.error(f"Failed to login: {e}")
            return 1

    # Download single file or entire repository
    if args.file:
        result = download_single_file(
            repo_id=args.repo_id,
            filename=args.file,
            local_dir=args.local_dir,
            repo_type=args.repo_type,
            token=args.token,
            revision=args.revision,
            force=args.force
        )
    else:
        result = download_repository(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            repo_type=args.repo_type,
            token=args.token,
            revision=args.revision,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
            cache_dir=args.cache_dir,
            force=args.force
        )

    if result:
        logger.info("Download completed successfully!")
        return 0
    else:
        logger.error("Download failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
