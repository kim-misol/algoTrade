import subprocess

try:
    # python run_binary_preprocessing.py <ticker> <trading_days> <windows>
    # "BBNI.JK" 종목을 학습 데이터 기간 20일로 설정하여 시가 대비 종가가 올랐는지 예측
    # 이미지 크기는 50x50으로 데이터 준비
    print(f'python run_binary_preprocessing.py "BBNI.JK" "20" "50"')
    subprocess.call(f'python run_binary_preprocessing.py  "BBNI.JK" "20" "50" ', shell=True)

    # python generatedata.py <pathdir> <origindir> <destinationdir>
    print(f'python generatedata.py "dataset" "20_50/BBNI.JK" "dataset_BBNIJK_20_50" ')
    subprocess.call(f'python generatedata.py "dataset" "20_50/BBNI.JK" "dataset_BBNIJK_20_50" ', shell=True)

    # python myDeepCNN.py -i <datasetdir> -e <numberofepoch> -d <dimensionsize> -b <batchsize> -o <outputresultreport>
    # 해당 신경망 알고리즘을 이용하여 저장된 이미지 데이터 경로 (datasetdir)로 전달하고
    # 전체 데이터 학습 횟수 (epoch)를 지정해 dimension size, 1 epoch를 몇 번에 걸쳐나눠 실행할지 batch size 값을 지정
    print(
        f'python myDeepCNN.py "-i" "dataset/dataset_BBNIJK_20_50" "-e" "50" "-d" "50" "-b" "8" "-o" "outputresult.txt"')
    subprocess.call(
        f'python myDeepCNN.py "-i" "dataset/dataset_BBNIJK_20_50" "-e" "50" "-d" "50" "-b" "8" "-o" "outputresult.txt"',
        shell=True)
except Exception as identifier:
    print(identifier)
