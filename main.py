print("Starting...")

from detect_spaghetti import detect
from time import sleep
from get_image import get_image
import discord
import asyncio

print("Imports complete.")


intents = discord.Intents.default()
intents.typing = False
intents.presences = False

token = "MTE0MTIwNDkxMzc3OTY1MDY0Mg.Gn-S0F.l5L2ixsATG53GgJR9uWabhbLWvYwTGlxVNOsEc"
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user.name}')

    log_channel = client.get_channel(1139710420471525527)
    counter = 0

    while True:
        if counter % 10 == 0:
            await log_channel.send(
                embed=discord.Embed(
                    title=f"Note: Monitoring for spaghetti. Has been running for {counter * 10} seconds.",
                    color=discord.Color.green(),
                )
            )

        image = get_image()
        if image:

            # image_path = 'O:\\Python Files\\Projects\\print-fail-detection\\fail_img.jpg'
            image_path = 'fail_img.jpg'

            with open(image_path, 'rb') as image_file:
                if counter % 60 == 0:
                    await log_channel.send(
                        embed=discord.Embed(
                            title=f"Note: Status, current camera image attached.",
                            color=discord.Color.green(),
                        ),
                        file=discord.File(image_file),
                    )

            boxes = detect(image, 0.6)
            if boxes:
                # pause()

                with open(image_path, 'rb') as image_file:
                    fail_message = await log_channel.send(
                        embed=discord.Embed(
                            title="FATAL: Spaghetti detected!",
                            description="Print automatically paused.",
                            color=discord.Color.red(),
                        ),
                        file=discord.File(image_file),
                    )

                await fail_message.add_reaction('👍')

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
                    title="Error: Failed to get image.",
                    color=discord.Color.orange(),
                )
            )

        counter += 1

        await asyncio.sleep(1)

client.run(token)
