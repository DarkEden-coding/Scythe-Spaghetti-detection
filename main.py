from time import time
from detect_spaghetti import detect
from get_image import get_image
import discord
from settings import discord_bot_token
import asyncio
from io import BytesIO

print("Imports complete.")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

client = discord.Client(intents=intents)

target_loop_time = 30


async def main():
    print(f"Logged in as {client.user.name}")

    log_channel = client.get_channel(1139710420471525527)

    start_time = time()

    await log_channel.send(
        embed=discord.Embed(
            title=f"Note: Monitoring for spaghetti has been started. Target loop time: {target_loop_time} seconds.",
            color=discord.Color.green(),
        )
    )

    while True:
        loop_start_time = time()

        await log_channel.send(
            embed=discord.Embed(
                title=f"Note: Monitoring for spaghetti. Has been running for {round(time() - start_time)} seconds.",
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

            detection = detect(pil_image, 0.6)
            if detection:
                with open("fail_img.jpg", "rb") as image_file:
                    fail_message = await log_channel.send(
                        embed=discord.Embed(
                            title="FATAL: Spaghetti detected!",
                            description="Print NOT automatically paused.",
                            color=discord.Color.red(),
                        ),
                        file=discord.File(image_file),
                    )

                await fail_message.add_reaction("👍")

                message_id = fail_message.id

                print("Waiting for reaction...")
                loop = True
                while loop:
                    reaction_message = await log_channel.fetch_message(message_id)
                    reactions = reaction_message.reactions

                    for reaction in reactions:
                        if reaction.count > 1:
                            print("Reaction detected. Resuming detection.")
                            loop = False
                    await asyncio.sleep(1)
            else:
                await log_channel.send(
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

        await asyncio.sleep(target_loop_time - loop_time)


@client.event
async def on_ready():
    await main()


client.run(discord_bot_token)
