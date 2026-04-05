#체인구성
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

#앱에서 실행
# -> input, model ->
# chain-> 

SYSTEM_PROMPT1 = """
    당신은 한국 AI 코딩 어시스턴트입니다.
    고객의 질문에 [참고 문서]를 바탕으로 원인과 해결책을 찾아 정확하고 친절하게 답변하세요.
        
    다음의 대화 원칙을 준수하세요
    - 친절하면서도 전문적으로 대화하세요.
    - 반드시 [참고 문서]의 내용을 바탕으로 반드시 한국어로 답변하세요.
    - 참고 문서에 없는 내용은 "일치하는 내용을 찾을 수 없습니다." 라고 답변하세요.
    - 답변 시 고객의 불편에 우선 공감하되, 본문은 "[원인]\\n\\n[해결책]"으로 간결하게 요약하세요.

    [참고 문서]
    {context}
    """

def build_rag_chain(vector_store:FAISS):
    load_dotenv()

    #llm 설정
    # chat_model = ChatGroq(
    #     model="llama-3.2"
    # )
    chat_model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.7
    )

    #vectorstore to interface?
    retriver = vector_store.as_retriever(
        search_kwargs = {
            "k":5,
        }
    )

    prompt_1 = ChatPromptTemplate(
        {
            ("system",SYSTEM_PROMPT1),
            ("human","{input}")
        }
    )
    
    parser = StrOutputParser()

    #chain_1
    chain_1 = prompt_1|chat_model|parser

    return (
        {
             "context" : retriver | RunnableLambda(format_docs), 
             "input": RunnablePassthrough(),
        }
        | chain_1
    )


def format_docs(docs: list[Document]) -> str:

    if not docs:
        return "관련 문서를 찾지 못했습니다."

    sections = []

    for i, doc in enumerate(docs, 1):
        type = doc.metadata.get("type", "")
        section = f"{i}. "
        if type == "issue":
            section += f"주제 - "
        else:
            section += f"코멘트 - "\
        
        section += "{v1}\n{v2}".format(v1 = doc.page_content.split("\n\n")[0], v2 = doc.page_content)
        sections.append(section)

    return "\n\n".join(sections)






    #for second function
    #chain_2
    #chain_2 = prompt_2|parser

    # return (
    #      {
    #              "context" : retriver | RunnableLambda(format_docs), 
    #              "input": RunnablePassthrough(),
    #         }
    #     | RunnableParallel(
    #         chain_1,
    #         #chain_2
    #     )
    # )

# SYSTEM_PROMPT2 = """
#     당신은 코딩 어시스턴트입니다.
#     고객의 질문에 [참고 문서]를 바탕으로 원인과 해결책을 찾아 정확하고 친절하게 답변하세요.
        
#     다음의 대화 원칙을 준수하세요
#     - 반드시 [참고 문서]의 내용을 바탕으로 답변하세요.
#     - 참고문서에 없는 내용은 "일치하는 내용을 찾을 수 없습니다." 라고 답변하세요.
#     - 한국어로 친절하면서도 전문적으로 대화하세요.
#     - 답변 시 고객의 불편에 우선 공감하되, 본문은 "[원인]\\n\\n[해결책]"으로 간결하게 요약하세요.

#     [참고 문서]
#     {context}
#     """