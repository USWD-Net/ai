from utils.merge import get_merged_data

# 원본 csv파일들 모여있는 폴더
RAW_DATA_PATH = "./ultrasonic/raw_data"

print('Loading Data . . .')
merged = get_merged_data(RAW_DATA_PATH)

# 라벨 초기화 및 정렬
print('Preparing Data . . .')
merged = merged.sort_values('timestamp')
merged["label"] = 0

print('=== LABEL TEXT FORM ===')
print('start end label')
with open("label.txt", "r") as f :
    labels = f.readlines()

for l in labels :
    start, end, label = l.replace('\n','').split()
    label = int(label)
    start, end = float(start), float(end)
    merged.loc[(merged['timestamp'] >= start) & (merged['timestamp'] <= end), 'label'] = label

merged.to_csv("./labeled/merged_labeled.csv",index=False)