import json
import pickle as pkl
import csv
import chardet
import openpyxl

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines

def write_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def read_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
        return data

def write_pkl(filepath, data):
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)
    print("write {}".format(filepath))

def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))

def read_csv(filepath, delimiter, skip_rows=0):
    '''
    default: can't filter the csv head
    :param filepath:
    :param delimiter:
    :return:
    '''
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = csv.reader(f, delimiter=delimiter)
        lines = [line for line in lines]
        return lines[skip_rows:]

def write_csv(filepath, data, delimiter):
    '''
    TSV is Tab-separated values and CSV, Comma-separated values
    :param data, is list
    '''
    with open(filepath, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=delimiter)
        csv_writer.writerows(data)

def read_xls(filepath, sheetname, skip_rows=0):
    '''
    :param filepath:
    :param sheetname: sheetname
    :return: list types of all instances
    '''
    workbook = openpyxl.load_workbook(filepath)
    # workbook.sheetnames
    booksheet = workbook[sheetname]
    rows = booksheet.rows
    all_rows = [r for r in rows]
    return all_rows[skip_rows:]
