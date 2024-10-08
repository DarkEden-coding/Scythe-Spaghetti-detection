# Scythe Spaghetti Detection Introduction

Hello, thanks for checking out my project! This is a **simple and lightweight** *OPEN SOURCE* and *SELF-HOSTED* 3D printer fail detection project. It is entirely based in Python and currently uses the **YOLOv8 object detection framework**. The project currently has a large model trained. If needed, I will try to train models for other sizes suitable for less powerful devices such as the Raspi-Zero. This project is in its early stages and is not yet ready for production use, so keep that in mind. If you find any issues please make an issue or pull request, any showcase of your usage of this project would be greatly appreciated, you can contact me using the info below. If the main branch does not have working code I deeply apologies, I am very new to the open source community and have some bad github habbits, if you see non working code on main please contact me asap!

## This project is in alpha! I apologise for any errors or dificulty you have with this project, but as a student I may not have the time to find all the errors, if you find any please report them in the Issues!

# !!!! LARGE UPDATE FOR RELIABILITY INCOMING SOON *tm!!!!
This update will fix many issues with the web requests and will hopefully reduce false positives and negitives in the detection system.

## Features:
- [x] **Supports any moonraker supported 3d printer**
- [x] **3D Printer Fail Detection**
- [x] **Discord Notifications and Status Updates**
- [x] **Lightweight and Easy to Use**
- [x] *Open Source*
- [x] *Self Hosted*
- [x] **Linux and Windows Support**
- [x] **YOLOv8 Object Detection**
- [x] *Large Model Trained*
- [x] Any Linux or Windows-based system support (theoretically, not yet tested, should only require Python support)
- [ ] XL Model Trained
- [ ] Medium Model Trained
- [ ] Small Model Trained
- [ ] Nano Model Trained
- [ ] Web Interface
- [ ] Email Notifications

## Slash Commands (not in main branch)(coming soon):
- /get_log_file this command will have the bot respond with your current log file, use this when submitting a bug report or when describing an error.
- /pause this command will pause the printer, used mainly if the bot does not detect errors but you see them in the status messages.

## Installation:
To install this software I reccomend using the terminal on the device that your are installing to, 
1. Clone the repository <br />
```bash
git clone "https://github.com/DarkEden-coding/Scythe-Spaghetti-detection.git"
```
2. Move into the installation folder
```bash
cd Scythe-Spaghetti-detection
```
3. Install the required packages <br />
```bash
pip install -r requirements.txt
```

## Configuration:
1. To configure your settings run settings_ui.py <br />
```bash
python3 settings_ui.py
``` 
follow the on screen instructions to configure the settings
2. to get the bot token, follow this guide: https://discordpy.readthedocs.io/en/stable/discord.html
3. To get the discord channel id and user id enable developer mode (settings -> advanced -> dev mode) in discord and right click the channel/user and click "copy id"

## Usage:
1. Run the `main.py` file <br />
```bash
python3 main.py
```
2. to make the bot automatically start on boot I recommend using a systemctl service, to do this you can use the following guide: https://www.raspberrypi-spy.co.uk/2015/10/how-to-autorun-a-python-script-on-boot-using-systemd/
<br /><br />
to check the status of the bot use 
```bash
systemctl status your-service-name
```

## Discord Message UI:
### status: <br />
![Alt text](https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/blob/master/readme_images/status.png?raw=true "Status Message")
<br />
### fail: <br />
![Alt text](https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/blob/master/readme_images/fail.png?raw=true "Fail Message")
## Defenetly 3d printer spaghetti, trust :) <br />
![Alt text](https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/blob/master/readme_images/Screenshot%202024-02-03%20at%201.58.17%20PM.png?raw=true "The Spaghett")

## Trouble Shooting:
### User permission issues:
If you encounter any user related issues try running the relevant file using sudo ie:
```bash 
sudo python3 main.py
```

## Contributing:
If you would like to contribute to the project, feel free to make a pull request. I will review it and merge it if it is suitable. If you have any questions, feel free to reach out to me on discord: `scytheeden` or email me at `darkedenc9@gmail.com`. If anyone is willing to "donate" gpu time to train new models that would be greatly appreciated.
# Relevant Links:
- [My Discord](https://discord.gg/users/806281289040396288)
- [YOLOv8](https://docs.ultralytics.com/)
- [Roboflow Dataset](https://universe.roboflow.com/dark-eden-nuheg/3d-printing-fail-detection)
# Notes:
- If you are a large company looking into using this software please contact me through email about possable modification to better suit your location and system.
- I am not responsable for any damage to your printer while using this project, use at your own risk, the detection will be as good as I can get it but if you have a spaghetti that is not detected and damages your printer I am not responsable.
