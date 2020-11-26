import os

import torch
from torch.utils import data

import numpy as np
import pandas as pd


def date2month(date: str):
    return int(date.split("-")[1])


class Dataset(data.Dataset):
    def __init__(self, csv_path):
        """Parse and store data."""
        super().__init__()
        data = pd.read_csv(csv_path)
        data = data.dropna().reset_index(drop=True)
        self.months = np.array(list(map(date2month, data["날짜"])))
        self.avg_temperatures = data["평균기온"].astype(float).values
        self.rainfalls = data["강수량"].astype(float).values
        self.fine_dust = data["미세먼지"].astype(float).values
        self.superfine_dust = data["초미세먼지"].astype(float).values
        data["holiday"] = ((data["주말공휴일"] == "1") | (data["주말공휴일"] == "토") | (data["주말공휴일"] == "일")).astype(int).values
        self.holiday = data["holiday"].astype(int).values
        self.spring = data["봄"].astype(int).values
        self.summer = data["여름"].astype(int).values
        self.autumn = data["가을"].astype(int).values
        self.winter = (1 - (data["봄"] + data["여름"] + data["가을"])).astype(int).values
        self.social_distance = data["사회적거리두기"].astype(int).values
        self.fog = data["안개"].astype(int).values
        self.visitors = data["방문객수"].astype(int).values
        self.len = len(data)

    def __getitem__(self, idx):
        return [self.months[idx], self.avg_temperatures[idx],
                self.rainfalls[idx], self.fine_dust[idx], self.superfine_dust[idx], self.holiday[idx],
                self.spring[idx], self.summer[idx], self.autumn[idx], self.winter[idx], 
                self.social_distance[idx], self.fog[idx]], self.visitors[idx]

    def __len__(self):
        return self.len


def get_loader(csv_path, batch_size, shuffle=True):
    dataset = Dataset(csv_path)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
    return dataset, data_loader
