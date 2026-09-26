"""Install the paper example into an unmodified, pinned upstream SkyRL checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess

UPSTREAM_COMMIT = "b2a08a0fc3f01f562d47db31a544a6fa1ce46bbe"


def install(checkout):
    checkout = Path(checkout).expanduser().resolve()
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream SkyRL commit {UPSTREAM_COMMIT}, got {commit}")
    dest = checkout / "skyrl-train/examples/userlm_paper"
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; remove the previous example before reinstalling")
    dest.mkdir(parents=True)
    for filename in ["env.py", "prompts.py", "main.py"]:
        shutil.copyfile(Path(__file__).parent / filename, dest / filename)
    (dest / "__init__.py").write_text("")
    print(f"Installed paper environment in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skyrl_dir", type=Path)
    install(parser.parse_args().skyrl_dir)
