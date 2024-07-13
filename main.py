import numpy as np

if __name__ == '__main__':
    result_1_path = "result/detection_1.npy"
    result_2_path = "result/detection_2.npy"
    result_path = "result/detection-20242202258.txt"

    result_1 = np.load(result_1_path)
    result_2 = np.load(result_2_path)

    timestamp = result_1[:, 0]
    label_1 = result_1[:, 1]
    label_2 = result_2[:, 1]
    label = label_1 | label_2
    print(sum(label))
    print(np.where(label == 1))

    result = np.stack((timestamp, label), axis=1)
    np.savetxt(result_path, result, fmt='%d')
