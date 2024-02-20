from settings import log_file
import os
import datetime

if not os.path.exists(log_file):
    with open(log_file, "w") as file:
        file.write("")

# write space and date to log file to indicate a new run
now = datetime.datetime.now()
with open(log_file, "a") as file:
    file.write("\n")
    file.write("-" * 100)
    file.write(f"\n{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    file.write("-" * 100)
    file.write("\n\n")


def log(message):

    # if file is over one gigabyte remove the first 100 lines
    if os.path.getsize(log_file) > 1000000000:
        with open(log_file, "r") as file:
            lines = file.readlines()
        with open(log_file, "w") as file:
            file.writelines(lines[100:])

    with open(log_file, "a") as file:
        file.write(f"{message}\n")
    print(message)
