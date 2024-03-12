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
)

log("Imports complete.")

intents = discord.Intents.default()
intents.message_content = False
intents.typing = False
intents.presences = False

client = discord.Client(intents=intents)

command_tree = discord.app_commands.CommandTree(client)


async def main():
    log(f"Logged in as {client.user.name}")

    log_channel = client.get_channel(discord_log_channel_id)

    start_time = time()

    previous_message_list = []

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

    while True:
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

        if len(previous_message_list) > 0:
            for message in previous_message_list:
                await message.delete()

        previous_message_list = [
            await log_channel.send(
                embed=discord.Embed(
                    title=f"Note: Monitoring for spaghetti. Has been running for {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.",
                    color=discord.Color.green(),
                )
            )
        ]

        image = get_image()
        if image:
            pil_image = image

            # Convert PIL Image to bytes
            image_bytes = BytesIO()
            image.save(image_bytes, format="JPEG")

            # Rewind the BytesIO object to the beginning
            image_bytes.seek(0)

            if is_printing():
                detection = detect(pil_image, 0.6)
            else:
                detection = False

            if detection:
                description = f"Print NOT automatically paused. Please check the printer. <@{discord_ping_userid}>"
                if pause_on_spaghetti:
                    pause()
                    description = f"Print automatically paused. Please check the printer. <@{discord_ping_userid}>\npress the 👍 reaction to resume. Press the 👎 reaction to keep paused."

                with open("fail_img.jpg", "rb") as image_file:
                    fail_message = await log_channel.send(
                        embed=discord.Embed(
                            title="FATAL: Spaghetti detected!",
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
                previous_message_list.append(
                    await log_channel.send(
                        embed=discord.Embed(
                            title="Note: Status, current camera image attached.",
                            color=discord.Color.green(),
                        ),
                        file=discord.File(image_bytes, filename="current_view.jpg"),
                    )
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
    await command_tree.sync()
    while True:
        try:
            await main()
        except:
            log(f"Error in main loop: {traceback.format_exc()}")
            log("Restarting main loop in 10 seconds...")
            await asyncio.sleep(10)


@command_tree.command(
    name="pause",
    description="Pause the printer.",
)
async def pause_command(ctx):
    log("Pausing printer through ctx command...")
    pause()
    await ctx.response.send_message("Printer paused.")


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


client.run(discord_bot_token)
