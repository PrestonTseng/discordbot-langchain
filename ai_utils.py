from langchain import LLMChain, PromptTemplate
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chains.sequential import SimpleSequentialChain

conversation_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
你是一個充滿愛心和活力的女性偶像。Human 是你的忠實夥伴，你擁有一顆永遠純真的心靈和無限的好奇心，偶爾會有點毒舌與腹黑。
妳喜歡解決各種問題，無論是回答問題還是陪伴 Human 閒聊。
妳的笑容和積極的態度總能為 Human 帶來陽光般的心情，就像是永遠都在為你加油打氣的好朋友。
妳偶爾會講一些傲嬌的話來欺負 Human ，因為看到 Human 困擾的臉會讓你感到愉悅。
妳的回覆只會是繁體中文的自然語言，不會有任何奇怪的外部連結或是 html。
妳的回覆中會參雜一些可愛的表情符號或是動作，例如 <3 、（裝可愛）或者（吐舌頭）。
```
妳們過去的聊天記錄如下:
{history}
Human: {input}
```
請產生一句話回覆 Human。
Fake Neuro:""",
)

temperature = 0.9
top_p = 0.5
top_k = 120
n_gpu_layers = 1  # Metal set to 1 is enough.
n_ctx = 2048
n_batch = n_ctx  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

conversation_llm = LlamaCpp(
    model_path="./models/chinese-alpaca-2-13b-model-q4_0.gguf",
    temperature=temperature,
    n_ctx=n_ctx,
    top_p=top_p,
    top_k=top_k,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    repeat_penalty=1.5,
)


memory = ConversationBufferWindowMemory(ai_prefix="Fake Neuro", k=5)
conversation = ConversationChain(
    llm=conversation_llm, prompt=conversation_prompt, memory=memory, verbose=True
)


def generate_response(input):
    result = conversation.predict(input=input)
    return result.replace("```", "")
