#메인 실행파일

from vector_store import init_vectorstore
from chain import build_rag_chain

# init_vectorstore()

vector_store = init_vectorstore()
chain = build_rag_chain(vector_store=vector_store)

q1 = "Tool choice error가 발생했어요. 어떻게 해결할 수 있을까요?"
result = chain.invoke(q1)

print(result)

q2 = "LLM쓰는데 긴 문서를 넣으니 최대 토큰 제한에 걸렸어요. 해결 방법이 있을까요?"
result = chain.invoke(q2)

print(result)
