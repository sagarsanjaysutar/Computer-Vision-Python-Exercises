import numpy as np

data = np.random.rand(5, 2)
labels = np.arange(5, 10, 1)
labels = np.reshape(labels, (5, 1))
data = np.concatenate((data, data))
labels = np.concatenate((labels, labels))
classes = {
    5: "cat",
    9: "dog",
    1: "cxat",
    2: "dxog"
}
# print(data, "\n")
# print(labels)
# new_labels = []
# new_data = []
# for label in labels:
#     if label[0] in classes.keys():
#         idx = (labels == label[0]).reshape(data.shape[0])
#         new_labels = np.append(new_labels, labels[labels == label[0]])
#         new_data = np.append(new_data, data[idx])


# new_data = np.reshape(new_data, (len(new_labels), 2))
# new_labels = np.reshape(new_labels, (len(new_labels), 1))
# print(new_data)
# print(new_labels)
our_classes = np.fromiter(classes.keys(), float)
our_classes = our_classes.reshape(our_classes.size, 1)

print(np.isin(labels, our_classes).flatten())
print(data[np.isin(labels, our_classes).flatten()])
print(labels[np.isin(labels, our_classes).flatten()])
