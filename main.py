import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import os

from ai_utils import generate_response

load_dotenv(find_dotenv())

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_message(message):
    if (
        message.stickers
        or message.author.bot
        or (
            message.reference
            and (
                message.reference.resolved.author != bot.user
                or message.reference.resolved.embeds
            )
        )
    ):
        return

    async with message.channel.typing():
        if message.mentions:
            for mention in message.mentions:
                message.content = message.content.replace(
                    f"<@{mention.id}>", f"這個名叫 {mention.display_name} 的人"
                )

        try:
            result = await asyncio.to_thread(generate_response, input=message.content)
            print(f"Human:{message.content}\nFake Neuro:{result}\n")
            await message.reply(result)
        except Exception as e:
            print(f"Error occurred: {e}")
            await message.channel.send("Sorry, I was unable to process your question.")


bot.run(os.environ.get("DISCORD_BOT_KEY"))
