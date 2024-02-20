import subprocess
from scythe_logging import log


def check_for_updates():
    """
    Check for updates to the repository and auto updates if settings allow
    :return: if the main.py file has been updated
    """
    try:
        # git fetch
        use_sudo = False
        fetch_result = subprocess.run(["git", "fetch"], capture_output=True, text=True)
        if (
            "Permission denied" in fetch_result.stderr
            or "detected dubious ownership" in fetch_result.stderr
        ):
            subprocess.run(
                ["sudo", "git", "fetch"],
                capture_output=True,
                text=True,
            )
            use_sudo = True

        # Run 'git status' to check the status of the local repository
        status_result = subprocess.run(
            ["sudo", "git", "status"] if use_sudo else ["git", "status"],
            capture_output=True,
            text=True,
        )

        # Check if there are updates available
        if "Your branch is behind" in status_result.stdout:
            log("\nThere are updates available. Auto Updating per settings...\n")
            log("Stashing local changes...")
            subprocess.run(["sudo", "git", "stash"] if use_sudo else ["git", "stash"])

            # Run 'git fetch' to fetch the latest changes from the remote repository
            log("Fetching latest changes...")
            subprocess.run(["sudo", "git", "fetch"] if use_sudo else ["git", "fetch"])

            # Run 'git pull' to pull the latest changes
            log("Pulling latest changes...")
            pull_result = subprocess.run(
                ["sudo", "git", "pull"] if use_sudo else ["git", "pull"],
                capture_output=True,
                text=True,
            )

            log(f"Pull result: {pull_result.stdout}")

            if "main.py" in pull_result.stdout:
                return True
        else:
            log("Your repository is up-to-date.")

    except Exception as e:
        log(f"An error occurred: {e} on line {e.__traceback__.tb_lineno}")


if __name__ == "__main__":
    check_for_updates()
