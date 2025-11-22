import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import (
    TextLoader, PythonLoader, CSVLoader, JSONLoader,
    Docx2txtLoader, UnstructuredPowerPointLoader,
    PyMuPDFLoader, UnstructuredMarkdownLoader,
    UnstructuredImageLoader, WebBaseLoader
)
_ = load_dotenv(find_dotenv())
client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

#数据存入向量库
from zhipuEmbedding import ZhipuAiEmbeddings
from langchain_community.vectorstores import Chroma

def dataLoadToVectordb(texts):
    embedding = ZhipuAiEmbeddings()
    persist_directory = 'E:/ai/llm-universe/data_base/vector_db/testchroma'
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=persist_directory
    )
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    return 

def get_file_paths(folder_path):
    #1.获取所有文件
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths[:3])

    # 下载所有文件并存储到text
    texts = []
    for file_path in file_paths:
        splitDocuments(file_path, texts)

    #2。清洗数据
    #去除多余换行，符号，空格等

    #3.文档数据分割
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 知识库中单段文本长度
    CHUNK_SIZE = 500

    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    docs = text_splitter.split_documents(texts)
    print(f"切分后的文件数量：{docs}")
    #print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in docs])}")

    #dataLoadToVectordb(docs)
    all_embeddding = []
    for i in range(0,len(docs),64):
        input_embeddings = docs[i : i + 64]
       #input_embeddings = [text.strip() for text in input_embeddings if text.strip()]
        dataLoadToVectordb(input_embeddings)

def splitDocuments(file_path, texts):
    file_type = file_path.split('.')[-1].lower()
    loader = None
    if file_type == 'pdf':
        loader = PyMuPDFLoader(file_path)  # PDF首选（高效稳定）
    elif file_type == 'md':
        loader = UnstructuredMarkdownLoader(file_path)  # Markdown
    elif file_type == 'txt':
        loader = TextLoader(file_path, encoding="utf-8")  # 纯文本
    elif file_type == 'py':
        loader = PythonLoader(file_path)  # Python代码
    elif file_type == 'csv':
        loader = CSVLoader(file_path, encoding="utf-8")  # 表格
    elif file_type == 'json':
        loader = JSONLoader(file_path, jq_schema=".content", text_content=False)  # JSON
    elif file_type == 'docx':
        loader = Docx2txtLoader(file_path) # Word（docx）
    elif file_type in ['xlsx', 'xls']:
        #loader = ExcelLoader(file_path)  # Excel（新旧格式）
         print(f"不支持的文件格式：{file_type} | 文件路径：{file_path}")
         return
    elif file_type in ['pptx', 'ppt']:
        loader = UnstructuredPowerPointLoader(file_path)  # PPT（新旧格式）
    elif file_type in ['png', 'jpg', 'jpeg']:
        loader = UnstructuredImageLoader(file_path)  # 图片（OCR提取）
    elif file_type == 'url':
        loader = WebBaseLoader(file_path)  # 普通网页
    elif file_type == 'epub':
        #loader = EpubLoader(file_path)  # 电子书
        print(f"不支持的文件格式：{file_type} | 文件路径：{file_path}")
        return
    else:
        print(f"不支持的文件格式：{file_type} | 文件路径：{file_path}")
        return

    if loader is not None:

        texts.extend(loader.load())



if __name__ == "__main__":
    get_file_paths("E:/ai/llm-universe/data_base/data")
