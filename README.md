# 퀀트 전략을 위한 인공지능 트레이딩

## Installation (TA-Lib)
1. [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) 사이트에서 
python 버전에 맞는 wheel file을 C드라이브에 다운로드 받는다. (C:\\)  
1. PyCharm Terminal 창에서 아래의 명령어를 실행하여 TA-Lib 설치
`pip install C:\\TA_Lib-0.4.19-cp38-cp38-win_amd64.whl`



-----------------

## 누구를 위한 책인가?

**퀀트 전략을 위한 인공지능 트레이딩** 책은 파이썬을 활용해서 금융데이터를 분석하는 방법을 알고싶은 사람들을 위한 책입니다.





## 주요 내용
파이썬을 활용한 금융 데이터 분석.

 - 전통적 분석(기술적 분석 + 재무제표 분석)

 - 머신러닝을 이용한 분석

 - 딥러닝을 이용한 분석

 - 백테스팅


 ## 아나콘다를 이용한 환경 설정 방법.


* 파이썬 3.6버전 가상환경 만들기.
```sh
(base) C:\Users\user-name> conda create -n py36 python=3.6
```
* 가상환경 활성화 For Windows
```sh
(base) C:\Users\user-name> activate py36
```

* 가상환경 활성화 For Mac
```sh
(base) $ source activate py36
```

* 패키지 전체 설치하기.
```sh
(py36) C:\Users\user-name> pip install -r requirements.txt
```

* 개별 패키지 설치하기.
```sh
(py36) C:\Users\user-name> pip install pandas
```

```sh
(py36) C:\Users\user-name> conda install pandas
```
* 특정 버전 패키지 설치하기.
```sh
(py36) C:\Users\user-name> conda install pandas==0.25.2
```


* (TIP) 패키지 내보내기.

```sh
(py36) C:\Users\user-name> pip freeze > requirements.txt
```

## 사용하는 딥러닝 버전.

 - tensorflow==1.15.0
 - Keras==2.2.4


 
