import os
import sys
from shutil import copyfile


def cre8outputdir(pathdir, targetdir):
    # create folder output
    if not os.path.exists("{}/{}".format(pathdir, targetdir)):
        os.mkdir("{}/{}".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train/0".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/0".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/train/1".format(pathdir, targetdir)):
        os.mkdir("{}/{}/train/1".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test/0".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/0".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/test/1".format(pathdir, targetdir)):
        os.mkdir("{}/{}/test/1".format(pathdir, targetdir))


pathdir = sys.argv[1]
origindir = sys.argv[2]
targetdir = sys.argv[3]

cre8outputdir(pathdir, targetdir)

# 데이터 불러오기
counttest = 0
counttrain = 0
# os.walk 함수로 현재 디렉터리의 파일과 하위 디렉터리를 순차적으로 순회
for root, dirs, files in os.walk(f"{pathdir}/{origindir}"):
    for file in files:

        tmp = root.replace('\\', '/')
        tmp_label = tmp.split('/')[-1]

        # 레이블이 0인 데이터를 해당 폴더에 원본을 보관하고, 복사본은 다른 경로로 이동시켜 추후 학습에 사용
        if tmp_label == '0':
            if 'test' in file:
                origin = f"{root}/{file}"
                destination = f"{pathdir}/{targetdir}/test/0/{file}"
                copyfile(origin, destination)
                counttest += 1
            elif 'train' in file:
                origin = f"{root}/{file}"
                destination = f"{pathdir}/{targetdir}/train/0/{file}"
                copyfile(origin, destination)
                counttrain += 1
        # 레이블이 1인 데이터를 해당 폴더에 원본을 보관하고, 복사본은 다른 경로로 이동시켜 추후 학습에 사용
        elif tmp_label == '1':
            if 'test' in file:
                origin = f"{root}/{file}"
                destination = f"{pathdir}/{targetdir}/test/1/{file}"
                copyfile(origin, destination)
                counttest += 1
            elif 'train' in file:
                origin = f"{root}/{file}"
                destination = f"{pathdir}/{targetdir}/train/1/{file}"
                copyfile(origin, destination)
                counttrain += 1

print(counttest)
print(counttrain)
