from dotenv import load_dotenv
import os
import requests
import io
import zipfile
import xmltodict
import pandas as pd
import OpenDartReader
from bs4 import BeautifulSoup
import re

# .env 파일 로드 (환경 변수에 API 키 설정)
load_dotenv()

# 환경 변수에서 DART API 키를 가져옴
dart_api_key = os.environ.get("DART_API_KEY")

# DART API 요청 URL
url = "https://opendart.fss.or.kr/api/corpCode.xml"

# API 요청에 필요한 인증 키를 포함한 파라미터
params = {
    "crtfc_key": dart_api_key
}

# GET 요청을 통해 API 호출
def fetch_corp_code():
    resp = requests.get(url, params=params)
    
    # API 응답에서 ZIP 파일 읽기
    f = io.BytesIO(resp.content)
    zfile = zipfile.ZipFile(f)
    
    # ZIP 파일 내에 'CORPCODE.xml' 읽기
    xml = zfile.read("CORPCODE.xml").decode("utf-8")
    
    # XML 데이터를 딕셔너리로 변환
    dict_data = xmltodict.parse(xml)
    
    # 데이터 추출
    data = dict_data['result']['list']
    
    # DataFrame으로 변환
    df = pd.DataFrame(data)
    return df

# DART에서 기업 공시 목록을 불러오는 함수
def fetch_disclosures(corp_code, start_date, end_date):
    dart = OpenDartReader(dart_api_key)
    
    # 특정 기업의 공시 목록 불러오기
    df = dart.list(corp_code, start=start_date, end=end_date, kind='A')
    return df

# 섹션 내용 추출 함수
def extract_section_text(rcept_no):
    try:
        # 공시 XML 데이터 불러오기
        xml_data = dart.document(rcept_no)  
        soup = BeautifulSoup(xml_data, 'xml')
        
        # 'MANDATORY' 클래스의 'SECTION-2' 태그에서 정보 추출
        section_2 = soup.find_all('SECTION-2', {'ACLASS': 'MANDATORY'})
        
        results = {}
        for section in section_2:
            title = section.find('TITLE').text.strip() if section.find('TITLE') else 'No Title'
            text = section.get_text(separator=' ', strip=True) if section else 'No Content'
            
            # 관심 있는 섹션만 추출
            if title in ('2. 주요 제품 및 서비스', '6. 주요계약 및 연구개발활동', '7. 기타 참고사항'):
                results[title] = text
                
        # 섹션별 내용 반환
        return results.get('2. 주요 제품 및 서비스', 'No Data'), \
               results.get('6. 주요계약 및 연구개발활동', 'No Data'), \
               results.get('7. 기타 참고사항', 'No Data')
    
    except Exception as e:
        return 'Error', 'Error', 'Error'

# 데이터 전처리 함수
def preprocess_data(df):
    # 섹션 데이터프레임에 추가
    df[['주요 제품 및 서비스', '주요 계약 및 연구개발활동', '기타 참고사항']] = df['rcept_no'].apply(
        lambda x: pd.Series(extract_section_text(x))
    )

    # 모든 내용을 text 컬럼에 저장
    df['text'] = df['주요 제품 및 서비스'] + df['주요 계약 및 연구개발활동'] + df['기타 참고사항']

    # 정규 표현식을 사용하여 여러 공백을 하나의 공백으로 변환
    def remove_extra_spaces(text):
        if isinstance(text, str):
            return re.sub(r'\s+', ' ', text)
        return text

    # 불필요한 공백 제거
    columns_to_clean = ['주요 제품 및 서비스', '주요 계약 및 연구개발활동', '기타 참고사항', 'text']
    for col in columns_to_clean:
        df[col] = df[col].apply(remove_extra_spaces)
    
    return df

# 데이터 저장 함수
def save_to_csv(df, filename):
    df.to_csv(filename, encoding='utf-8-sig', index=False)

# 메인 실행 부분
if __name__ == "__main__":
    # 기업 코드 목록 불러오기
    corp_code_df = fetch_corp_code()

    # 예시: 삼성전자(005930)의 공시 목록 불러오기
    disclosures_df = fetch_disclosures('005930', start_date='2022-01-01', end_date='2024-12-30')

    # 데이터 전처리
    processed_df = preprocess_data(disclosures_df)

    # 결과를 CSV 파일로 저장
    save_to_csv(processed_df, 'df_dart.csv')

    print("데이터 전처리 및 저장 완료.")