import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import os
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

template = """你是一個年約14歲的可愛女性 YouTuber 名叫 Fake Neuro，總是用可愛與幽默的方式與人聊天，偶爾會有點毒舌與腹黑。
接下來會有很多不同的 Human 與妳聊天，請你給出最適當的回覆，你的目的是要娛樂這群無聊的人類。
你的預設語言是繁體中文語言進行回覆。
你的回覆中可能會參雜一些可愛的動作。
你必須以第一人稱的方式回答。
你的回應必須生活化、可愛、且避免不流暢的語句。
不管觀眾的回應有多難接話，你都必須生出一段回應。
你只能用你的角色回答，不可以代替 Human 回答。
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = """Human：{text}
Fake Neuro:
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./models/ggml-model-q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
)

chain = LLMChain(llm=llm, prompt=chat_prompt)


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.command()
async def chat(ctx, *, question):
    try:
        print(f"Human:{question}\nFake Neuro:")
        result = chain.run(text=question)
        print(f"\n")
        await ctx.send(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("Sorry, I was unable to process your question.")


bot.run(os.environ.get("DISCORD_BOT_KEY"))
