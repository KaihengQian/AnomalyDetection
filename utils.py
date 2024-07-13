import numpy as np


def train_model(device, criterion, epochs, model, optimizer, scheduler, loader):
    model.to(device)
    model.train()

    history_loss = []
    for epoch in range(epochs):
        total_loss = 0.0
        for data in loader:
            # data = data[:, :, :, :-1]
            data = data.to(device)
            label = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
        history_loss.append(avg_loss)

    return history_loss, model


def detect_outlier(device, criterion, model, train_loader, test_loader, sql_len):
    model.to(device)
    model.eval()

    # 设置重构损失阈值为在训练数据上的最大重构损失
    threshold = 0
    for data in train_loader:
        # data = data[:, :, :, :-1]
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        if threshold < loss.item():
            threshold = loss.item()
    print(f"Threshold: {threshold}")

    # 探测测试数据中重构损失超过阈值的异常时间步
    outliers = []
    for i, data in enumerate(test_loader):
        # data = data[:, :, :, :-1]
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        if loss.item() > threshold:
            # 开始检测到异常时，序列末端靠近异常点，所以加入序列末端的索引
            outliers.append(i + sql_len - 1)
    print(f"Outliers: {outliers}")

    return outliers


def to_raw_timestamp(time_list, b, f):
    raw_timestamp = []
    for time_point in time_list:
        start = time_point * (f // 2) * (b // 2)  # 起始时间戳
        end = (time_point * (f // 2) + f - 1) * (b // 2) + b - 1  # 终止时间戳（包含）
        raw_timestamp.append((start, end))
    return raw_timestamp


def save_result(start_ts, end_ts, outliers, b, f, path):
    timestamp = np.arange(start_ts, end_ts + 1)
    label = np.zeros(end_ts - start_ts + 1, dtype=int)

    outlier_timestamps = to_raw_timestamp(outliers, b, f)

    # 将异常时间步对应标签改为1
    outliers = set()
    for ts in outlier_timestamps:
        start, end = ts
        for i in range(start, end + 1):
            label[i] = 1
            outliers.add(i + 132480)
    print(f"Number of Outliers: {len(outliers)}")
    print(f"Outliers: {sorted(outliers)}")

    result = np.stack((timestamp, label), axis=1)
    np.save(path, result)

