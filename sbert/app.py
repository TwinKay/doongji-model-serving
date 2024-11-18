from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# Milvus 연결
connections.connect("default", host="localhost", port="19530")
collection = Collection("danji_embeddings")

# 임베딩 모델 로드
model_name = "jhgan/ko-sroberta-multitask"
embedder = SentenceTransformer(model_name)

# 요청 데이터 모델 정의
class QueryModel(BaseModel):
    query: str
    top_k: int = 16384

@app.post("/find_similar")
async def find_similar(data: QueryModel):
    try:
        # 입력 쿼리 임베딩 생성
        query_embedding = embedder.encode(data.query, convert_to_tensor=False).tolist()

        # Milvus에서 검색
        search_params = {"metric_type": "IP"}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=data.top_k,
            output_fields=["danji_id"]
        )

        # 검색 결과에서 danji_id와 유사도 추출
        result_list = []
        for result in results[0]:
            danji_id = result.entity.get("danji_id")
            similarity = result.distance
            result_list.append({"danji_id": danji_id, "similarity": similarity})

        return {"results": result_list}
    except Exception as e:
        return {"error": str(e)}

# 8001번 포트에서 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
