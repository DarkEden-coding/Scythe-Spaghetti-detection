import subprocess
import platform
from settings import enable_auto_update


def check_for_updates():
    try:
        # Use 'sudo' on non-Windows platforms
        git_command = "git"
        if platform.system() != "Windows":
            git_command = "sudo git"

        # Run 'git status' to check the status of the local repository
        status_result = subprocess.run(
            [git_command, "status"], capture_output=True, text=True
        )
        print(status_result.stdout)

        # Check if there are updates available
        if "Your branch is behind" in status_result.stdout:
            if enable_auto_update:
                print("\nThere are updates available. Auto Updating per settings...\n")
                print("Stashing local changes...")
                subprocess.run([git_command, "stash"])

                # Run 'git fetch' to fetch the latest changes from the remote repository
                print("Fetching latest changes...")
                subprocess.run([git_command, "fetch"])

                # Run 'git pull' to pull the latest changes
                print("Pulling latest changes...")
                subprocess.run([git_command, "pull"])
            else:
                print("There are updates available. Please pull the latest changes.")
        else:
            print("Your repository is up-to-date.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    check_for_updates()
