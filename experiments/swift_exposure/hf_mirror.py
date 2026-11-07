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
        name = os.path.basename(out_dir.rstrip("/"))
        api.upload_folder(folder_path=out_dir, path_in_repo=name, repo_id=repo, delete_patterns=[f"{name}/student_latest.pt"] if not os.path.exists(os.path.join(out_dir, "student_latest.pt")) else None)
        print(f"uploaded {out_dir} -> hf://{repo}", flush=True)
        return True
    except Exception as e:
        print("HF upload failed:", repr(e)[:300], flush=True)
        return False


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        hf_upload(d)
