from collections import Counter
import random
import numpy as np

# def most_common_integer(lst):
#     if not lst:
#         return None
#     counter = Counter(lst)
#     most_common = counter.most_common(1)[0][0] 
#     return most_common

def get_first_non_zero_occurrence_over_ten(number_list: list) -> int:
    counts = Counter()
    for num in number_list:
        if num != 0:
            counts[num] += 1
            if counts[num] >= 10:
                return num
    return 0 

def get_lbl(df) :
    lbls = df['label'].to_list()
    return get_first_non_zero_occurrence_over_ten(lbls)


def minmax(data: list[list[int]]) -> list[list[float]]:
    if not data or not data[0]:
        return []
    np_data = np.array(data, dtype=np.float64)

    actual_min = 0
    actual_max = 1023
    scaled_np_data = (np_data - actual_min) / (actual_max - actual_min)
    return scaled_np_data.tolist()

def split_list(data, ratio):
    shuffled = data[:]
    random.shuffle(shuffled)
    i = int(len(shuffled) * ratio)
    splited = shuffled[:i]
    remain = shuffled[i:]
    return splited, remain