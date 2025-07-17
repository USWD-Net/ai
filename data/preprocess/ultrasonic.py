from utils.preprocess import get_lbl, minmax, split_list
import pandas as pd
import pickle

# timestamp | s1 | s2 | s3 | s4 | label
# prerpocessing -> window size로 split -> 겹치는 구간?
# 만약 label 0이 아닌게 절반 이상 많으면 -> label 지정
# pickle 파일 형식 -> dict {label1 : ... , label2 : [1dim array[ ... ], [ ... ]]}
# 0.1초에 하나, 1초에 10개, 10초에 100개 1분에 600개, 10분에 6000개, 1시간에 36000개 24시간에 864000개 ... 하루에 약 350만개 데이터?
# 전처리하면 2배 -> 하루에 700만개 데이터
# label 0인것 중에서는 몇개만 extract 해서 사용

labeled_df = pd.read_csv("./ultrasonic/labeled/merged_labeled.csv")

WINDOW_SIZE = 100
STEPS = 50
TEST_RATIO = 0.2
VAL_RATIO = 0.1

out_dict = {
}

idx_last = len(labeled_df) - WINDOW_SIZE

print(f"Total Data Length : {len(labeled_df)}")
print(f"Extracted Windows : {len(range(0, idx_last, STEPS))}")
print(f"Extracted Data Length : {len(range(0, idx_last, STEPS))*WINDOW_SIZE*4}")

for start in range(0, idx_last, STEPS) :
    end = start + WINDOW_SIZE
    df_tmp = labeled_df.iloc[start:end,:]
    sdata = [df_tmp['s1'].to_list(), df_tmp['s2'].to_list(), df_tmp['s3'].to_list(), df_tmp['s4'].to_list()]
    lbl = get_lbl(df_tmp)
    if lbl not in list(out_dict.keys()) :
        out_dict[lbl] = []
        
    sdata = minmax(sdata)
    out_dict[lbl].append(sdata)

test_dict = {
}

val_dict = {
}

train_dict = {
}


for k in sorted(list(out_dict.keys())) :
    windows_len = len(out_dict[k])
    print(f"Label {k} windows : {windows_len} , data length : {windows_len*WINDOW_SIZE*4}")
    test_data, tv_data = split_list(out_dict[k], TEST_RATIO)
    val_data, train_data = split_list(tv_data, VAL_RATIO)
    test_dict[k] = test_data
    train_dict[k] = train_data
    val_dict[k] = val_data
    
with open("../dataset/ultrasonic/train.pkl", "wb") as f :
    pickle.dump(train_dict, f)
    
with open("../dataset/ultrasonic/val.pkl", "wb") as f :
    pickle.dump(val_dict, f)
    
with open("../dataset/ultrasonic/test.pkl", "wb") as f :
    pickle.dump(test_dict, f)
    
# dict {0 : [[[s1 ...], [s2 ...], [s3 ...], [s4 ...]] , [[s1 ...] [s2 ...], [s3 ...], [s4 ...]] ... ] }