import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
from dotenv import load_dotenv, find_dotenv
from zhipuLLM import ZhipuaiLLM
from zhipuEmbedding import ZhipuAiEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

#ui
import gradio as gr


_ = load_dotenv(find_dotenv())
api_key=os.environ["ZHIPUAI_API_KEY"]

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


def show_switch_status(switch_state):
    return switch_state

# 在文件顶部定义转换函数
def format_chat_history(chatbot):
    """将 Gradio Chatbot 格式转为 LangChain 支持的 chat_history 格式"""
    formatted_history = []
    for human_msg, ai_msg in chatbot:
        formatted_history.append(("human", human_msg))
        formatted_history.append(("ai", ai_msg))
    return formatted_history

def chatbot_response(input, chatbot, isUseRAG):
    """根据开关状态返回提示信息"""

    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=api_key)
    if isUseRAG:

        # 问答链的系统prompt
        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        # 制定prompt template
        qa_prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
         # 定义问答链
        qa_chain = (
            RunnablePassthrough.assign(context=combine_docs) # 使用 combine_docs 函数整合 qa_prompt 中的 context
            | qa_prompt # 问答模板
            | llm
            | StrOutputParser() # 规定输出的格式为 str
        )

        #
        #获取得到向量库
        vectordb = Chroma(
            persist_directory='E:/ai/llm-universe/data_base/vector_db/chroma',  # 允许我们将persist_directory目录保存到磁盘上
            embedding_function=ZhipuAiEmbeddings()
        )
        #取数据
        retriever = vectordb.as_retriever(search_kwargs={"k": 1})

        
        # 压缩问题的系统 prompt
        condense_question_system_template = (
            "请根据聊天记录完善用户最新的问题，"
            "如果用户最新的问题不需要完善则返回用户的问题。"
            )
        # 构造 压缩问题的 prompt template
        condense_question_prompt = ChatPromptTemplate([
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ])
        
        retrieve_docs = RunnableBranch(
        # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
        condense_question_prompt | llm | StrOutputParser() | retriever,
        )
    
        # 定义带有历史记录的问答链
        qa_history_chain = RunnablePassthrough.assign(
            context = (lambda x: x) | retrieve_docs # 将查询结果存为 content
            ).assign(answer=qa_chain)

        result = qa_history_chain.invoke({
            "input": input,
            "chat_history": format_chat_history(chatbot)
        })

        print(result)
        chatbot.append((input,result["answer"]))
        return [chatbot,input]
    else:
        result = llm.invoke(input)
        print(chatbot)
        chatbot.append((input,result.content))
        return [chatbot,input]

# 创建界面
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🤖 RAG 智能聊天机器人
    支持直接调用大模型或结合本地知识库（RAG）回答问题
    """)

    chatbot = gr.Chatbot(
        label="对话历史",
        height=500,  # 对话框高度
        avatar_images=(None, "https://gradio.s3-us-west-2.amazonaws.com/guides/robot.png")  # （可选）用户/机器人头像
    )

    with gr.Row():
        chebox = gr.Checkbox(
            label="RAG",
            value=False
        )

    with gr.Row():
        input = gr.Textbox(
            label="输入你的问题",
            placeholder="例如：",
            lines=2,
            container=False
        )
        submit_btn = gr.Button("发送", variant="primary", icon="📤")

    submit_btn.click(
        fn=chatbot_response,
        inputs=[input, chatbot, chebox],  # 输入：用户消息 + 历史对话 + 开关状态
        outputs=[chatbot, input]  # 输出：更新后的对话 + 清空输入框
    )
    

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,  # 显示错误信息（调试用）
    )

    
    
     
    
