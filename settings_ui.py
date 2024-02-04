default_settings = {
    "discord_bot_token": "YOUR DISCORD BOT TOKEN",
    "discord_ping_userid": 111111111111111111,
    "discord_log_channel_id": 111111111111111111,
    "target_loop_time": 30,
    "printer_url": "http://mainsailos.local/",
    "webcam_name": "Bed",
    "pause_on_spaghetti": True,
    "use_cuda": False,
}

# Prompt the user for each setting
settings = {}
for key, value in default_settings.items():
    user_input = input(
        f"Please enter your desired setting for: {key} (default: {value}): "
    )
    # Use the user input if it's not empty, otherwise use the default value, if the default is not string then convert the user input to the same type
    if user_input:
        settings[key] = type(value)(user_input)
    else:
        settings[key] = value

# Save the settings to settings.py
with open("settings.py", "w") as file:
    for key, value in settings.items():
        file.write(f"{key} = {repr(value)}\n")

print("\nSettings dump: ")
print(settings)

print("\nSettings saved to settings.py")
