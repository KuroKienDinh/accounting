#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    konto_db_writer.py
# @Author:      kien.tadinh
# @Time:        4/1/2024 11:50 PM
import os

import pandas as pd
import pyodbc
from tqdm import tqdm


class KontoDBWriter:
    def __init__(self, data_path, server, database, username, password):
        self.data_path = data_path
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection = pyodbc.connect(
            f'DRIVER={{SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}')
        self.cursor = self.connection.cursor()

    def delete_data_by_group(self, group_nr):
        sql = "DELETE FROM [dbo].[group_chart_acounts] WHERE [groupnr] = ?"
        try:
            self.cursor.execute(sql, group_nr)
            self.cursor.commit()
            # print(f"Data for group {group_nr} has been deleted from the database")
        except Exception as e:
            print(e)

    def is_exist(self, group_nr):
        sql = "SELECT COUNT(*) FROM [dbo].[group_chart_acounts] WHERE [groupnr] = ?"
        try:
            self.cursor.execute(sql, group_nr)
            count = self.cursor.fetchone()[0]
            if count > 0:
                return True
            return False
        except Exception as e:
            print(e)
            return False

    def write_to_db(self, data):
        sql = "INSERT INTO [dbo].[group_chart_acounts]([groupnr],[accountnr],[account_name],[standard_chart_account_id],[total_items]) VALUES (?,?,?,?,?)"
        try:
            self.cursor.executemany(sql, data)
            self.cursor.commit()
        except Exception as e:
            print(e)

    def disconnect(self):
        self.cursor.close()
        self.connection.close()


def process_group(data_path, group_nr):
    # db.delete_data_by_group(group_nr)
    if db.is_exist(group_nr) or os.path.exists(os.path.join(data_path, group_nr, f"{group_nr}.konto.parquet")) is False:
        return
    konto_df = pd.read_parquet(os.path.join(data_path, group_nr, f"{group_nr}.konto.parquet"))
    if not konto_df.empty:
        konto_df = konto_df[['konto', 'bezeichnung', 'items']]
        konto_df.dropna(subset=['bezeichnung'], inplace=True)
        konto_data = konto_df.values.tolist()
        params = [(group_nr, record[0], record[1], None, int(record[2])) for record in konto_data]
        db.write_to_db(params)


if __name__ == "__main__":

    data_path = "../data/model_Progros_50"
    SERVER = '192.168.227.4'
    DATABASE = 'ConnectAccounting'
    USERNAME = 'dev.accounting'
    PASSWORD = 'qweasd'

    db = KontoDBWriter(data_path, SERVER, DATABASE, USERNAME, PASSWORD)
    for group_nr in tqdm(os.listdir(data_path), desc="Processing groups"):
        process_group(data_path, group_nr)
    db.disconnect()
