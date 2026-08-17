import subprocess
from pathlib import Path

#python scripts/create_tiff_from_cub.py
# ============================================================
# USER SETTING
# ============================================================

SOURCE_CUB = Path("path/file.cub")


# ============================================================
# SCRIPT
# ============================================================

def run_command(command):
    """
    Run a shell command and show output if it fails.
    """
    print("\nRunning command:")
    print(" ".join(str(c) for c in command))

    result = subprocess.run(
        [str(c) for c in command],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}"
        )

    return result


def main():
    if not SOURCE_CUB.exists():
        raise FileNotFoundError(f"Could not find CUB file: {SOURCE_CUB}")

    output_tif = SOURCE_CUB.with_suffix(".tif")

    print(f"Source CUB: {SOURCE_CUB}")
    print(f"Output TIFF: {output_tif}")

    # First check that ISIS can read the cube label.
    run_command([
        "catlab",
        f"from={SOURCE_CUB}"
    ])

    # Convert CUB to TIFF preview.
    run_command([
        "isis2std",
        f"from={SOURCE_CUB}",
        f"to={output_tif}",
        "format=tiff"
    ])

    print("\nDone.")
    print(f"Created TIFF preview: {output_tif}")


if __name__ == "__main__":
    main()
