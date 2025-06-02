import pandas as pd
import os

# 디렉토리 경로 설정
data_dir = "mixed"  # mixed 폴더 경로

# train과 test 데이터를 저장할 빈 리스트 생성
train_files = []
test_files = []

# mixed 디렉토리 내의 파일 목록 확인
for filename in os.listdir(data_dir):
    if filename.startswith("MixedSet_train_Fold") and filename.endswith(".csv"):
        train_files.append(filename)
    elif filename.startswith("MixedSet_test_Fold") and filename.endswith(".csv"):
        test_files.append(filename)

# 파일 정렬 (Fold0부터 Fold9까지 순서대로)
train_files.sort()
test_files.sort()

# train 데이터 합치기
train_dfs = []
for train_file in train_files:
    file_path = os.path.join(data_dir, train_file)
    df = pd.read_csv(file_path)
    train_dfs.append(df)

# 모든 train 데이터를 하나의 DataFrame으로 병합
combined_train_df = pd.concat(train_dfs, ignore_index=True)

# test 데이터 합치기
test_dfs = []
for test_file in test_files:
    file_path = os.path.join(data_dir, test_file)
    df = pd.read_csv(file_path)
    test_dfs.append(df)

# 모든 test 데이터를 하나의 DataFrame으로 병합
combined_test_df = pd.concat(test_dfs, ignore_index=True)

# 결과 저장
combined_train_df.to_csv("MixedSet_train_combined.csv", index=False)
combined_test_df.to_csv("MixedSet_test_combined.csv", index=False)

# 결과 확인
print(f"Combined train dataset size: {len(combined_train_df)}")
print(f"Combined test dataset size: {len(combined_test_df)}")