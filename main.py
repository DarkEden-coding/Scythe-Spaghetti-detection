from scytheautoupdate import check_for_updates
import os
import subprocess

if not os.path.exists("settings.py"):
    subprocess.run(["python", "settings_ui.py"])

from scythe_logging import log

if check_for_updates():  # Check for updates
    log("Changes have been made to main.py. Restarting the program...")
    import sys

    subprocess.run([sys.executable, "main.py"])
    sys.exit()

from time import time
from detect_spaghetti import detect
from web_interaction.get_image import get_image
from web_interaction.is_printing import is_printing
from web_interaction.pause_print import pause
from web_interaction.resume_print import resume
import discord
import asyncio
from io import BytesIO
import traceback

from settings import (
    discord_bot_token,
    discord_ping_userid,
    discord_log_channel_id,
    target_loop_time,
    pause_on_spaghetti,
    detection_threshold,
)

log("Imports complete.")

intents = discord.Intents.default()
intents.message_content = False
intents.typing = False
intents.presences = False

client = discord.Client(intents=intents)

command_tree = discord.app_commands.CommandTree(client)

running = True


async def delete_message(message):
    """
    Delete messages robustly
    :param message: the message to delete
    :return:
    """
    for _ in range(5):
        try:
            await message.delete()
            break
        except:
            log("Error deleting message, retrying...")
            await asyncio.sleep(0.5)


async def main():
    global running

    log(f"Logged in as {client.user.name}")

    log_channel = client.get_channel(discord_log_channel_id)

    start_time = time()

    previous_time_message = None
    previous_status_message = None

    await log_channel.send(
        embed=discord.Embed(
            title="-" * 50,
            color=discord.Color.green(),
        )
    )
    await log_channel.send(
        embed=discord.Embed(
            title=f"Note: Monitoring for spaghetti has been started. Target loop time: {target_loop_time} seconds.",
            color=discord.Color.green(),
        )
    )

    while running:
        loop_start_time = time()

        # string of the current run time, days, hours, and minutes
        run_time = time() - start_time
        days = run_time // (24 * 3600)
        run_time = run_time % (24 * 3600)
        hours = run_time // 3600
        run_time %= 3600
        minutes = run_time // 60
        run_time %= 60
        seconds = run_time

        if previous_time_message:
            await delete_message(previous_time_message)

        previous_time_message = await log_channel.send(
            embed=discord.Embed(
                title=f"Note: Monitoring for spaghetti. Has been running for {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.",
                color=discord.Color.green(),
            )
        )

        image = get_image()
        if image:
            pil_image = image

            # Convert PIL Image to bytes
            image_bytes = BytesIO()
            image.save(image_bytes, format="JPEG")

            # Rewind the BytesIO object to the beginning
            image_bytes.seek(0)

            if is_printing():
                detection = detect(pil_image, detection_threshold)
            else:
                detection = False

            if detection:
                description = "Print NOT automatically paused. Please check the printer.\npress the 👍 reaction to resume detection. Press the 👎 reaction to pause the print."
                if pause_on_spaghetti:
                    pause()
                    description = "Print automatically paused. Please check the printer.\npress the 👍 reaction to resume detection. "

                with open("fail_img.jpg", "rb") as image_file:
                    fail_message = await log_channel.send(
                        content=f"<@{discord_ping_userid}>",
                        embed=discord.Embed(
                            title=f"FATAL: Spaghetti detected!",
                            description=description,
                            color=discord.Color.red(),
                        ),
                        file=discord.File(image_file),
                    )

                await fail_message.add_reaction("👍")
                if not pause_on_spaghetti:
                    await fail_message.add_reaction("👎")

                message_id = fail_message.id

                log("Waiting for reaction...")

                loop = True
                while loop:
                    reaction_message = await log_channel.fetch_message(message_id)
                    reactions = reaction_message.reactions

                    for reaction in reactions:
                        if reaction.emoji == "👍" and reaction.count > 1:
                            log("Reaction detected. Resuming detection.")
                            loop = False
                            break

                        if not pause_on_spaghetti:
                            if reaction.emoji == "👎" and reaction.count > 1:
                                log("Reaction detected. Pausing.")
                                pause()
                                loop = False
                                break

                    await asyncio.sleep(0.1)
            else:
                if previous_status_message:
                    await delete_message(previous_status_message)

                previous_status_message = await log_channel.send(
                    embed=discord.Embed(
                        title="Note: Status, current camera image attached.",
                        color=discord.Color.green(),
                    ),
                    file=discord.File(image_bytes, filename="current_view.jpg"),
                )

        else:
            await log_channel.send(
                embed=discord.Embed(
                    title="Error: Failed to get image.",
                    color=discord.Color.orange(),
                )
            )

        loop_end_time = time()
        loop_time = loop_end_time - loop_start_time

        if loop_time > target_loop_time:
            log(
                f"Loop time exceeded target loop time by {loop_time - target_loop_time} seconds."
            )
            continue

        if is_printing():
            log(f"Loop time: {loop_time} seconds.")
        await asyncio.sleep(target_loop_time - loop_time)


@client.event
async def on_ready():
    global running
    await command_tree.sync()
    while True:
        try:
            await main()
        except:
            log(f"Error in main loop: {traceback.format_exc()}")
            log("Restarting main loop in 30 seconds...")
            running = False
            await asyncio.sleep(30)
            running = True


@command_tree.command(
    name="pause",
    description="Pause the printer.",
)
async def pause_command(ctx):
    log("Pausing printer through ctx command...")
    pause()
    await ctx.response.send_message("Printer paused.")


@command_tree.command(
    name="resume",
    description="Resume the printer.",
)
async def resume_command(ctx):
    log("Resuming printer through ctx command...")
    resume()
    await ctx.response.send_message("Printer resumed.")


@command_tree.command(
    name="get_log_file",
    description="Get the log file.",
)
async def get_log_file(ctx):
    with open("spaghetti.log", "rb") as file:
        log("Sending log file to user...")
        await ctx.response.send_message(
            "Log file:", file=discord.File(file, filename="spaghetti.log")
        )


@command_tree.command(
    name="get_printer_status",
    description="Get the printer status.",
)
async def get_printer_status(ctx):
    log("Sending printer status to user...")
    await ctx.response.send_message(f"Printer printing: {is_printing()}")


@command_tree.command(
    name="get_image",
    description="Get the current image.",
)
async def get_image_command(ctx):
    image = get_image()
    if image:
        log("Sending image to user...")
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        await ctx.response.send_message(
            "Current image:", file=discord.File(image_bytes, filename="current_view.jpg")
        )
    else:
        log("Failed to get image.")
        await ctx.response.send_message("Failed to get image.")


@command_tree.command(
    name="test_ping",
    description="send a test ping.",
)
async def test_ping(ctx):
    log("Sending test ping...")
    # send ping as regular message
    await ctx.response.send_message(f"<@{discord_ping_userid}>")


@command_tree.command(
    name="upload_detection_model",
    description="Upload a new detection model from a URL."
)
async def upload_detection_model(ctx, url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = url.split('/')[-1]  # Extract filename from URL
            with open("model_utils/largeModel.onnx", "wb") as f:
                f.write(response.content)
            await ctx.send(f"Model file '{filename}' downloaded successfully.")
            log("New detection model successfully downloaded.")
        else:
            await ctx.send("Failed to download the file from the provided URL.")
            log("User attempted to upload a detection model, but the download failed.")
    except Exception as e:
        await ctx.send(f"An error occurred: {e}")
        log(f"An error occurred while downloading the detection model: {e}")


client.run(discord_bot_token)
