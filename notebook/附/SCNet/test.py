import os
from dotenv import load_dotenv, find_dotenv
from zhipuLLM import ZhipuaiLLM
from zhipuEmbedding import ZhipuAiEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

#ui
import gradio as gr


_ = load_dotenv(find_dotenv())
api_key=os.environ["ZHIPUAI_API_KEY"]

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

'''
def get_completion(prompt, model="glm-4-plus", temperature=1):
    messages = [{"role": "user", "content": prompt}]
    res = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    if len(res.choices) > 0:
        print(res.choices[0].message.content)
        return res.choices[0].message.content
    return "generate answer error"
'''

def show_switch_status(switch_state):
    return switch_state
    
def chatbot_response(input, chatbot, isUseRAG):
    """根据开关状态返回提示信息"""

    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=api_key)
    if isUseRAG:

        #获取得到向量库
        vectordb = Chroma(
            persist_directory='E:/ai/llm-universe/data_base/vector_db/testchroma',  # 允许我们将persist_directory目录保存到磁盘上
            embedding_function=ZhipuAiEmbeddings()
        )
        #取数据
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        #合并数据
        combiner = RunnableLambda(combine_docs)
        retrieval_chain = retriever | combiner

        #构建链
        template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {input}
        """
        # 将template通过 PromptTemplate 转为可以在LCEL中使用的类型
        prompt = PromptTemplate(template=template)

        qa_chain = (
            RunnableParallel(  {"context": retrieval_chain, "input": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )

        result = qa_chain.invoke(input)
        print(result)
        chatbot.append((input,result))
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
        bubble_full_width=False,  # 气泡不占满宽度（更美观）
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
        server_port=7861,
        server_name="127.0.0.1",
        share=False,
        show_error=True  # 显示错误信息（调试用）
    )

    
    
     
    
