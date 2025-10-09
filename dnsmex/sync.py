import os
import subprocess
import shutil
from datetime import datetime


def sync(source_paths, destination="."):
    """Syncs multiple source paths to the destination while preserving history by
    backing up files that will be overwritten, changed, or deleted.

    Args:
        source_paths (str): Multi-line string with source paths, one per line.
        destination (str): The destination directory to sync the files to.
    """
    backup_dir = ".sync-backup"

    # Ensure the backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Create a timestamped backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, timestamp)
    os.makedirs(backup_subdir)

    # Function to run rsync and capture output
    def run_rsync(source, destination, dry_run=True):
        cmd = ["rsync", "-av", "--delete"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.append(source)
        cmd.append(destination)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    # Process each source path
    for source in source_paths.strip().split("\n"):
        source = source.strip()

        # Run dry-run rsync and collect affected files
        dry_run_output = run_rsync(source, destination, dry_run=True)
        affected_files = []

        # Print the dry-run output and ask for confirmation
        print(dry_run_output)
        if any(
            line.startswith(("<f", ">f", "cd", "deleting"))
            for line in dry_run_output.splitlines()
        ):
            confirm = input("Do you want to continue? [y/N] ")
            if confirm.lower() != "y":
                print("Sync aborted.")
                return

        # Parse dry-run output to identify files to be backed up
        for line in dry_run_output.splitlines():
            if line.startswith("deleting "):
                file_path = line.replace("deleting ", "").strip()
                affected_files.append(file_path)
            elif (
                line.startswith(">f") or line.startswith("cd") or line.startswith("<f")
            ):
                file_path = line.split()[-1].strip()
                affected_files.append(file_path)

        # Backup affected files
        for file_path in affected_files:
            source_path = os.path.join(destination, file_path)
            backup_path = os.path.join(backup_subdir, file_path)

            # Ensure the directory structure exists in the backup directory
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # Move or copy the file to the backup directory if it exists
            if os.path.exists(source_path):
                shutil.move(source_path, backup_path)

        # Run the actual rsync
        run_rsync(source, destination, dry_run=False)

    # If the backup directory is empty, remove it
    if not os.listdir(backup_subdir):
        os.rmdir(backup_subdir)
