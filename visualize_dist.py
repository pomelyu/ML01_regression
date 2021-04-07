import numpy as np

# selected_index = list(range(40)) + [57, 75]
# selected_index = list(range(40)) + [41, 59, 77, 42, 60, 78, 43, 61, 79, 44, 62, 80, 58, 76]
selected_index = [41, 59, 77, 42, 60, 78, 43, 61, 79, 44, 62, 80, 58, 76]

train_data = np.loadtxt("data/covid.train.csv", delimiter=",", skiprows=1)[:, 1:-1][:, selected_index]
valid_index = np.array(range(0, len(train_data), 10))
valid_data = train_data[valid_index]
train_data = train_data[[i for i in range(len(train_data)) if i not in valid_index]]
test_data = np.loadtxt("data/covid.test.csv", delimiter=",", skiprows=1)[:, 1:][:, selected_index]

mean = np.mean(train_data, axis=0, keepdims=True)
std = np.mean(train_data, axis=0, keepdims=True)

print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

train_data[:, 40:] = (train_data[:, 40:] - mean[:, 40:]) / std[:, 40:]
valid_data[:, 40:] = (valid_data[:, 40:] - mean[:, 40:]) / std[:, 40:]
test_data[:, 40:] = (test_data[:, 40:] - mean[:, 40:]) / std[:, 40:]


all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
all_label = np.concatenate([
    np.ones(len(train_data)) * 0,
    np.ones(len(valid_data)) * 1,
    np.ones(len(test_data)) * 2,
], axis=0)

print(all_data.shape)
print(all_label.shape)

np.savetxt("all_data.txt", all_data, "%.4f", delimiter="\t")
np.savetxt("all_label.txt", all_label, "%d")
