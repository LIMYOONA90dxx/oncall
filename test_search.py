import sys
sys.path.insert(0, '.')
from app.services.vector_search_service import vector_search_service

try:
    results = vector_search_service.search_similar_documents('测试', top_k=3)
    print(f'找到 {len(results)} 个结果')
    for r in results:
        print(f'- {r.content[:50]}...')
except Exception as e:
    import traceback
    traceback.print_exc()
