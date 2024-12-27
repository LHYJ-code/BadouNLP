#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

#待切分文本
sentence = "经常有意见分歧"


def all_cut(sentence, Dict):
    result = []
    length = len(sentence)

    def backtracking(index, segments):
        if index == length:
            result.append(segments.copy())
            return
        for end in range(index + 1, length + 1):
            word = sentence[index:end]
            if word in Dict:
                new_segments = segments.copy()
                new_segments.append(word)
                backtracking(end, new_segments)
                new_segments.pop()

    backtracking(0, [])
    return result


if __name__ == "__main__":
    target = all_cut(sentence, Dict)
    for item in target:
        print(item)
