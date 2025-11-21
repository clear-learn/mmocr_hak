import openai
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_book_spine_by_word(image_path, api_key):
    base64_image = encode_image(image_path)

    prompt = """
다음 이미지는 책의 책등을 촬영한 것입니다.
글자들이 세로로 쓰여 있고, 글자 간격이 좁으며 폰트나 디자인 요소로 인해 OCR로 인식이 어렵습니다.
이미지 전처리를 통해 텍스트를 최대한 수동으로 식별해주세요.

**요청 사항**:
- 글자 방향은 세로입니다.
- 이미지 전처리를 적용하여 (회전, 대비 증가  등) 가능한 최대한 명확히 보이도록 처리해주세요.
- OCR은 사용하지 않고 사람이 수동으로 인식하는 방식으로 해주세요.
- **텍스트는 문장 단위로 나누고, 각 문장을 단어 단위로 구분하여 아래 표로 정리해주세요.**
- **단어 단위로 단어의 가운데 좌표를 측정해주세요.
- 이때 텍스트에 가장 가운데의 좌표여야합니다.
- 좌표는 이 이미지의 해상도가 (82x816)인걸 감안해서 정확히 측정해주세요
- y값이 정확해야해 x는 비슷할테니
- 회전을 진행할꺼니까 가로기준으로 x,y를 하자 그래야 정확하지

**출력 형식 예시(표로 작성해주세요)**:

{
  "annotations": [
    {
      "center": [x,y],
      "id": 1,
      "text": "권성근 장군 회고록",
      "image_id": 1
    },
    {
      "center": [x,y],
      "id": 2,
      "text": "조국을 위해",
      "image_id": 1
    },
    {
      "center": [x,y],
      "id": 3,
      "text": "하늘을 날다",
      "image_id": 1
    }
  ]
}

이 이미지를 바탕으로 위 요구에 따라 단어와 위치를 인식해주세요.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
            ]}
        ],
        "max_tokens": 1500
    }

    response = openai.ChatCompletion.create(**payload)
    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    api_key = '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'
    base_image_dir = "image"
    # 폴더를 재귀 탐색하며 이미지(.jpg, .jpeg, .png 등) 찾기
    for root, dirs, files in os.walk(base_image_dir):
        for file in files:
            # 파일 확장자로 이미지 판별
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)