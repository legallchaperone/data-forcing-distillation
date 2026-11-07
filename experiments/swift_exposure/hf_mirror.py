"""Mirror a run directory to a private Hugging Face repo when HF_REPO is set (token from `hf auth login` or HF_TOKEN).
Never raises: an upload failure must not kill a finished GPU job."""
import os


def hf_upload(out_dir, repo=None):
    repo = repo or os.environ.get("HF_REPO")
    if not repo or not os.path.isdir(out_dir):
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=out_dir, path_in_repo=os.path.basename(out_dir.rstrip("/")), repo_id=repo)
        print(f"uploaded {out_dir} -> hf://{repo}", flush=True)
        return True
    except Exception as e:
        print("HF upload failed:", repr(e)[:300], flush=True)
        return False


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        hf_upload(d)
