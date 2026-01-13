import logging

import torch


def collate_shapenet_stability(list_data):
    B = len(list_data)
    list_data = [data for data in list_data if data is not None]
    if B != len(list_data):
        logging.info(f"Retain {len(list_data)} from {B} data.")
    if len(list_data) == 0:
        raise ValueError("No data in the batch")

    pcds, rots, labels = list(zip(*list_data))

    ret = dict()
    ret["batch_size"] = len(rots)
    ret["pcd"] = torch.vstack(pcds)
    ret["rots"] = torch.vstack(rots)
    ret["label"] = torch.LongTensor(labels)

    return ret