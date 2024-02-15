import subprocess
import platform
from settings import enable_auto_update


def check_for_updates():
    try:
        # Use 'sudo' on non-Windows platforms
        os = True
        if platform.system() != "Windows":
            os = False

        # git fetch
        subprocess.run(["git", "fetch"] if os else ["sudo", "git", "fetch"])

        # Run 'git status' to check the status of the local repository
        status_result = subprocess.run(
            ["git", "status"] if os else ["sudo", "git", "status"],
            capture_output=True,
            text=True,
        )

        # Check if there are updates available
        if "Your branch is behind" in status_result.stdout:
            if enable_auto_update:
                print("\nThere are updates available. Auto Updating per settings...\n")
                print("Stashing local changes...")
                subprocess.run(["git", "stash"] if os else ["sudo", "git", "stash"])

                # Run 'git fetch' to fetch the latest changes from the remote repository
                print("Fetching latest changes...")
                subprocess.run(["git", "fetch"] if os else ["sudo", "git", "fetch"])

                # Run 'git pull' to pull the latest changes
                print("Pulling latest changes...")
                subprocess.run(["git", "pull"] if os else ["sudo", "git", "pull"])
            else:
                print("There are updates available. Please pull the latest changes.")
        else:
            print("Your repository is up-to-date.")

    except Exception as e:
        print(f"An error occurred: {e} on line {e.__traceback__.tb_lineno}")


if __name__ == "__main__":
    check_for_updates()
