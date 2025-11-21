from pymilvus import MilvusClient


def search_books_by_title(
    query_text,
    collection_name="domestic_book_meta_embedding",
    milvus_uri="http://10.10.13.129:19530",
    limit=10
):
    """
    제목(title)으로 책 검색

    Args:
        query_text: 검색할 텍스트 (예: "해리포터")
        collection_name: 컬렉션 이름
        milvus_uri: Milvus 서버 URI
        limit: 반환할 결과 수
    """

    # Milvus 클라이언트 생성
    client = MilvusClient(uri=milvus_uri)

    print("=" * 80)
    print(f"🔎 검색 쿼리: '{query_text}'")
    print(f"   컬렉션: {collection_name}")
    print(f"   검색 필드: itemTitle_embedding")
    print(f"   결과 수: {limit}")
    print("=" * 80)
    print()

    # Milvus Function을 사용한 검색
    # anns_field에 임베딩 필드를 지정하고, data에 텍스트를 넣으면
    # Milvus가 자동으로 Function을 호출하여 임베딩 생성
    results = client.search(
        collection_name=collection_name,
        data=[query_text],  # 검색할 텍스트 (Milvus Function이 자동으로 임베딩 생성)
        anns_field="itemTitle_embedding",  # 검색할 벡터 필드
        limit=limit,
        output_fields=[
            "itemId",
            "itemTitle",
            "itemSubTitle",
            "authorName",
            "publisherName",
            "price",
            "custReviewRank",
            "custReviewCount"
        ]
    )

    # 결과 출력
    print(f"📚 검색 결과 (상위 {limit}개):\n")

    if not results or not results[0]:
        print("   검색 결과가 없습니다.")
        return

    for i, hit in enumerate(results[0], 1):
        entity = hit['entity']
        distance = hit['distance']

        print(f"   {i}. {entity.get('itemTitle', 'N/A')}")
        if entity.get('itemSubTitle'):
            print(f"      부제: {entity.get('itemSubTitle')}")
        print(f"      저자: {entity.get('authorName', 'N/A')}")
        print(f"      출판사: {entity.get('publisherName', 'N/A')}")
        print(f"      가격: {entity.get('price', 0):,}원")

        if entity.get('custReviewRank'):
            print(f"      평점: {entity.get('custReviewRank'):.1f}/5.0 ({entity.get('custReviewCount', 0)}개 리뷰)")

        print(f"      유사도 점수: {distance:.4f}")
        print(f"      Item ID: {entity.get('itemId')}")
        print()


def search_books_by_author(
    query_text,
    collection_name="domestic_book_meta_embedding",
    milvus_uri="http://10.10.13.129:19530",
    limit=10
):
    """
    저자(author)로 책 검색

    Args:
        query_text: 검색할 저자명 (예: "J.K. 롤링")
        collection_name: 컬렉션 이름
        milvus_uri: Milvus 서버 URI
        limit: 반환할 결과 수
    """

    # Milvus 클라이언트 생성
    client = MilvusClient(uri=milvus_uri)

    print("=" * 80)
    print(f"🔎 검색 쿼리: '{query_text}'")
    print(f"   컬렉션: {collection_name}")
    print(f"   검색 필드: authorName_embedding")
    print(f"   결과 수: {limit}")
    print("=" * 80)
    print()

    # authorName_embedding으로 검색
    results = client.search(
        collection_name=collection_name,
        data=[query_text],
        anns_field="authorName_embedding",
        limit=limit,
        output_fields=[
            "itemId",
            "itemTitle",
            "itemSubTitle",
            "authorName",
            "authorNameOriginal",
            "publisherName",
            "price"
        ]
    )

    # 결과 출력
    print(f"📚 검색 결과 (상위 {limit}개):\n")

    if not results or not results[0]:
        print("   검색 결과가 없습니다.")
        return

    for i, hit in enumerate(results[0], 1):
        entity = hit['entity']
        distance = hit['distance']

        print(f"   {i}. {entity.get('itemTitle', 'N/A')}")
        print(f"      저자: {entity.get('authorName', 'N/A')}")
        if entity.get('authorNameOriginal'):
            print(f"      원저자명: {entity.get('authorNameOriginal')}")
        print(f"      출판사: {entity.get('publisherName', 'N/A')}")
        print(f"      가격: {entity.get('price', 0):,}원")
        print(f"      유사도 점수: {distance:.4f}")
        print(f"      Item ID: {entity.get('itemId')}")
        print()


def search_books_with_filter(
    query_text,
    price_min=None,
    price_max=None,
    collection_name="domestic_book_meta_embedding",
    milvus_uri="http://10.10.13.129:19530",
    limit=10
):
    """
    필터링을 포함한 책 검색

    Args:
        query_text: 검색할 텍스트
        price_min: 최소 가격
        price_max: 최대 가격
        collection_name: 컬렉션 이름
        milvus_uri: Milvus 서버 URI
        limit: 반환할 결과 수
    """

    # Milvus 클라이언트 생성
    client = MilvusClient(uri=milvus_uri)

    # 필터 조건 생성
    filter_conditions = []
    if price_min is not None:
        filter_conditions.append(f"price >= {price_min}")
    if price_max is not None:
        filter_conditions.append(f"price <= {price_max}")

    filter_expr = " and ".join(filter_conditions) if filter_conditions else None

    print("=" * 80)
    print(f"🔎 검색 쿼리: '{query_text}'")
    print(f"   컬렉션: {collection_name}")
    print(f"   검색 필드: itemTitle_embedding")
    if filter_expr:
        print(f"   필터: {filter_expr}")
    print(f"   결과 수: {limit}")
    print("=" * 80)
    print()

    # 검색
    results = client.search(
        collection_name=collection_name,
        data=[query_text],
        anns_field="itemTitle_embedding",
        filter=filter_expr,
        limit=limit,
        output_fields=[
            "itemId",
            "itemTitle",
            "authorName",
            "publisherName",
            "price",
            "custReviewRank",
            "custReviewCount"
        ]
    )

    # 결과 출력
    print(f"📚 검색 결과 (상위 {limit}개):\n")

    if not results or not results[0]:
        print("   검색 결과가 없습니다.")
        return

    for i, hit in enumerate(results[0], 1):
        entity = hit['entity']
        distance = hit['distance']

        print(f"   {i}. {entity.get('itemTitle', 'N/A')}")
        print(f"      저자: {entity.get('authorName', 'N/A')}")
        print(f"      출판사: {entity.get('publisherName', 'N/A')}")
        print(f"      가격: {entity.get('price', 0):,}원")
        if entity.get('custReviewRank'):
            print(f"      평점: {entity.get('custReviewRank'):.1f}/5.0")
        print(f"      유사도 점수: {distance:.4f}")
        print()


def main():
    """메인 함수 - 다양한 검색 예제"""

    print("\n" + "=" * 80)
    print("📖 Domestic Book Meta 검색 테스트")
    print("=" * 80)
    print()

    # 1. 제목으로 검색: 해리포터
    # print("\n[ 테스트 1: 제목으로 '해리포터' 검색 ]\n")
    search_books_by_title("나는 치사 은퇴하고 싶다.", limit=5)

    print("\n" + "=" * 80 + "\n")

    # 2. 제목으로 검색: 한국 역사
    # print("\n[ 테스트 2: 제목으로 '한국 역사' 검색 ]\n")
    # search_books_by_title("한국 역사", limit=5)
    #
    # print("\n" + "=" * 80 + "\n")
    #
    # # 3. 저자로 검색
    # print("\n[ 테스트 3: 저자로 '김영하' 검색 ]\n")
    # search_books_by_author("김영하", limit=5)
    #
    # print("\n" + "=" * 80 + "\n")
    #
    # # 4. 가격 필터링을 포함한 검색
    # print("\n[ 테스트 4: '소설' 검색 + 가격 10,000원 ~ 20,000원 ]\n")
    # search_books_with_filter("소설", price_min=10000, price_max=20000, limit=5)
    #
    # print("\n" + "=" * 80 + "\n")
    # print("✅ 검색 테스트 완료!")


if __name__ == "__main__":
    main()
