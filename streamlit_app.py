#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1ZrZ78z_1T9nte1qYuaZPv3foAwtkaELH'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 업로드한 이미지")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 해당 인종과 관련된 문화 영상")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://mblogthumb-phinf.pstatic.net/MjAxNzExMTJfMTQ4/MDAxNTEwNDY1ODcxNDgz.v7GQYdnCTq7TKZ4C2fg9RFT4xhS7A-0XEu_6Ml_bIp8g.rl9T4ZoLunqmL6_BVmMmFr-tSoSwfSL7Uk7_CUCGEBEg.PNG.chjc9/%EC%82%AC%EC%A7%841.png?type=w800",
            "https://blog.kakaocdn.net/dn/cFavul/btsHmG6SsBn/Ibpv5hhvi1ZWt8liBvh4Pk/img.webp",
            "https://i.namu.wiki/i/0NFzmGwF4yVTuV5O64FP2RZy8Uwlu6houJvyMTslwCZzl87pbKct36mbSEsTdHlDV6E-aVqXa5-gf994TnTxBQ.webp"
        ],
        'videos': [
            "https://youtu.be/Rqza4_o_dRg?si=WnR6Q7pI7ia6frk3",
            "https://youtu.be/aUA9ziIyWwU?si=itTQ5RIWKJDywncv",
            "https://youtu.be/SAdWoYiNURk?si=QAL6bT09-xPluebb"
        ],
        'texts': [
            "치파오의 역사",
            "치파오의 현황",
            "치파오 만드는 방법"
        ]
    },
    labels[1]: {
        'images': [
            "https://imgcp.aacdn.jp/img-a/600/auto/global-aaj-front/article/2018/02/5a96a97597e70_5a96a3c1c1347_209397231.jpg",
            "https://i.namu.wiki/i/MEg9hKDLxkcWnaBtzmq9S_LNkmeeOyq_PwMNctcf6hLbB1FbNlVRwzNLoE1197aSdCPWR-LeYpflBVOwP5uTuw.webp",
            "https://i.namu.wiki/i/AwbzlQcSWKLtpFyQzS6UcvfRa3zhcXdSQOkuqPGpC_d5u6miggbMkdSDEH_ZXdU2kSfWGitkpASQPY1pnRE33g.webp"
        ],
        'videos': [
            "https://youtu.be/RZPPHzIlB1s?si=9IaFclihO1E8zabi",
            "https://youtu.be/-fJ8YKnt-ZM?si=SPt8e-jj9j_IYln2",
            "https://youtu.be/0q0xPmVUKNU?si=GbeYPvLVHsI5dajC"
        ],
        'texts': [
            "기모노의 역사",
            "기모노의 현황",
            "기모노 만드는 방법"
        ]
    },
    labels[2]: {
        'images': [
            "https://image.thehyundai.com/static/3/7/5/74/A1/40A1745736_0_600.jpg",
            "https://www.iwedding.co.kr/center/iweddingb/product/800_co_sl_k076_11928_1607323729_30410800_3232256099.jpg",
            "https://m.hanboknam.com/web/product/big/202109/2a5d60c6a45aa35fb17976a1d372d330.jpg"
        ],
        'videos': [
            "https://youtu.be/M9gZyBtu7-w?si=A3GBweelgkEZZYyD",
            "https://youtu.be/3e4oMGMZ5t8?si=tAAn7sINmR6__Z5b",
            "https://youtu.be/gkKFgOttxD8?si=aYVG7xvTJgUfFs9S"
        ],
        'texts': [
            "한복의 역사",
            "한복의 현황",
            "한복 만드는 방법"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

