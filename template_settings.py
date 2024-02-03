discord_bot_token = "YOUR DISCORD BOT TOKEN"
discord_ping_userid = 111111111111111111  # your discord user id
discord_log_channel_id = 111111111111111111  # your discord log channel id

target_loop_time = 30  # how often to check for spaghetti in seconds (will try to be as close to this as possible given the time it takes to process the image)

printer_url = "http://mainsailos.local/"  # the url/ip of your printer
pause_on_spaghetti = (
    False  # set to True if you want the printer to pause when spaghetti is detected
)

use_cuda = False  # set to True if you have a CUDA compatible GPU and want to use it for the spaghetti detection
