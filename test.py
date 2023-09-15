import discord

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

token = "MTE0MTIwNDkxMzc3OTY1MDY0Mg.Gn-S0F.l5L2ixsATG53GgJR9uWabhbLWvYwTGlxVNOsEc"
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user.name}')

    logs_channel = client.get_channel(1139710420471525527)

    target_message = await logs_channel.fetch_message(1148427869782020116)  # Replace with the actual message ID

    # Get the reactions on the message
    reactions = target_message.reactions

    print(reactions)


client.run(token)

