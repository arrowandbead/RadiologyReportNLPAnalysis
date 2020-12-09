# from openpyxl import load_workbook
import os
import pandas as pd
import re
import math
import numpy as np
import collections
import unicodedata

# def iterateThroughFilesInFolder(folderPath):
#     return os.listdir(folderPath)

# def getFieldsFromXlsx(filePath, fields):
#     workbook = load_workbook(filePath)
#     sheet1 = workbook.get_sheet_by_name("clinical")
#     iter = sheet1.iter_rows()
#     row0 = [cell.value for cell in next(iter)]
#     fieldToIndexMap = {}
#     for index, thing in enumerate(row0):
#         fieldToIndexMap[thing] = index
#     outPutList = []
#     for row in iter:
#         newDict = {}
#         rowAsList = [cell.value for cell in row]
#         for thing in fields:
#             newDict[thing] = rowAsList[fieldToIndexMap[thing]]
#         outPutList.append(newDict)
#     return outPutList


# def getExtractEvaluationInformation(filePath):
#     workbook = load_workbook(filePath)
#     sheet1 = workbook.get_sheet_by_name("sheet1")
#     iter = sheet1.iter_rows()
#     row0 = [cell.value for cell in next(iter)]

#     indexToFieldMap = {}
#     for index, thing in enumerate(row0):
#         indexToFieldMap[index] = thing
#     indexToFieldMap[2] = "time"

#     outPutList = []
#     for row in iter:
#         patientReportList = []
#         outPutList.append(patientReportList)
#         rowAsList = list([cell.value for cell in row])
#         currReportDict = None
#         for index,thing in enumerate(rowAsList):
#             if(index == 0 or thing == None):
#                 continue
#             if "post-MR" in indexToFieldMap[index]:
#                 if currReportDict != None:
#                     if "cancer status" not in currReportDict and "evidence of cancer" not in currReportDict:
#                         patientReportList = patientReportList[:-1]
#                 currReportDict = {}
#                 patientReportList.append(currReportDict)
#                 currReportDict["post-MR"] = str(thing).strip(" ").split(',')
#             else:

#                 if thing != "" and "#" not in str(thing):
#                     currReportDict[indexToFieldMap[index]] = str(thing)
#         if currReportDict != None:
#             if "cancer status" not in currReportDict and "evidence of cancer" not in currReportDict:
#                 patientReportList = patientReportList[:-1]
#     return outPutList

# OPTIONAL TODO: Remove list numbers from impressions in numbered lists

def get_report_labels():
    """
        Input: None
        Output: dictionary of {report ID : evidence of cancer} from 'evaluation of reports.xlsx'
    """
    data = pd.read_excel("../data/evaluation of reports.xlsx")
    report_labels = {}

    # loop through each column in the excel sheet of report ids - evidence of cancer
    for i in range(1, 21):
        report_col = "post-MR "
        report_col += str(i)
        evidence_col = "evidence of cancer" 
        if i != 1:
            # pandas adds a ".number" to each repeated column name
            evidence_col += "."
            evidence_col += str(i - 1)
        reports = data[report_col].astype(str)
        labels = data[evidence_col].astype(float)
        temp = dict(zip(reports, labels))
        report_labels.update(temp)

    # remove reports that don't have a label
    report_labels = remove_nan_labels(report_labels)
    return report_labels
    
def get_report_impressions():
    """
        Input: None
         Output: dictionary of {report ids : impressions} from /reports directory
    """
    report_impressions = {}
    for filename in os.scandir("../data/reports"):
        name = filename.name
        if (name == ".DS_Store"):
            continue
        tokenized_name = re.split('_|-', name)
        report_id = tokenized_name[-4]
        if (list(tokenized_name[-1])[-3:] != ['t','x','t']): # temporary -- will only look at .txt's
            continue
        # print("--------------------------------------")
        # print(open("../data/reports/" + name, "r").read())
        report = open("../data/reports/" + name, "r").readlines()
        add_to_impression = False
        impression = ""
        for line in report:
            tokenize_line = line.split()
            if (len(tokenize_line) == 0):
                # blank line
                continue
            first_word = tokenize_line[0].strip()
            if first_word == "END":
                # end of impression
                add_to_impression = False
            if add_to_impression:
                impression += line.strip() + " "
            if first_word == "IMPRESSION" or first_word == "IMPRESSION:":
                impression = ""
                first_line = tokenize_line[1:]
                for word in first_line:
                    impression += word + " "
                add_to_impression = True
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("Impression extracted:")
        # print(impression)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        report_impressions.update({report_id:impression})
    return report_impressions

def remove_nan_labels(label_dictionary):
    """
        Input: dictionary of {report id:label or evidence of cancer}
        Output: same dictionary, with reports without labels (represented as not a number values) removed
    """
    reports_without_labels = []
    for report_id in label_dictionary:
        if math.isnan(label_dictionary[report_id]):
            reports_without_labels.append(report_id)
    for report_id in reports_without_labels:
        del label_dictionary[report_id]
    return label_dictionary

def clean_dictionary_entries():
    """
        Input: None
        Output: tuple of lists (impressions, labels) for each impression and its corresponding label
    """
    impressions_dict = get_report_impressions()
    labels_dict = get_report_labels()

    # ensure report id keys are all in the same format, start with sorted key list
    impressions_sorted_old = sorted(impressions_dict)
    labels_sorted_old = sorted(labels_dict) 

    # remove unicode characters
    impressions_sorted_new = [unicodedata.normalize("NFKD", x) for x in impressions_sorted_old] 
    labels_sorted_new = [unicodedata.normalize("NFKD", x) for x in labels_sorted_old]

    # remove whitespaces to even out input
    impressions_sorted_new = ["".join(x.split()) for x in impressions_sorted_new]
    labels_sorted_new = ["".join(x.split()) for x in labels_sorted_new]

    # make all keys integers (change ids that look like 1.0 -> 1)
    impressions_sorted_new = make_entries_integers(impressions_sorted_new)
    labels_sorted_new = make_entries_integers(labels_sorted_new)

    # update dictionaries with formatted keys
    for i in range(len(impressions_dict)):
        impressions_dict[impressions_sorted_new[i]] = impressions_dict.pop(impressions_sorted_old[i])

    for i in range(len(labels_dict)):
        labels_dict[labels_sorted_new[i]] = labels_dict.pop(labels_sorted_old[i])

    # find keys outside the intersection of both sets
    to_remove_from_impressions, to_remove_from_labels = cross_check_sets(impressions_sorted_new, labels_sorted_new)

    # remove reports outside the intersection of both dictionaries from each dictionary
    for key in to_remove_from_impressions:
        impressions_sorted_new.remove(key)
        del impressions_dict[key]
    
    for key in to_remove_from_labels:
        labels_sorted_new.remove(key)
        del labels_dict[key]

    assert(impressions_dict.keys() == labels_dict.keys())

    sorted_impressions = list(collections.OrderedDict(sorted(impressions_dict.items())).values())
    sorted_labels = list(collections.OrderedDict(sorted(labels_dict.items())).values())

    # turn labels back to ints
    sorted_labels = [int(i) for i in sorted_labels]

    return (sorted_impressions, sorted_labels)


def cross_check_sets(impressions_set, labels_set):
    """
        Input: set of report_ids in the {report_ids:impressions} and {report_ids:labels} dictionaries, respectively
        Output: dictionary of {report ID : evidence of cancer} from 'evaluation of reports.xlsx'
    """
    impressions_set = set(impressions_set)
    labels_set = set(labels_set)
    to_remove_from_impressions = [x for x in impressions_set if x not in labels_set]
    to_remove_from_labels = [x for x in labels_set if x not in impressions_set]
    return (to_remove_from_impressions, to_remove_from_labels)

def make_entries_integers(key_list):
    for i in range(len(key_list)):
        report_ids = key_list[i].split(",")
        for j in range(len(report_ids)):
            # if number/report id is 1.0, it will change to 1
            try:
                report_ids[j] = str(int(float(report_ids[j])))
            except:
                pass
        key_list[i] = "".join(report_ids)
    return key_list

def get_data():
    impressions, labels = clean_dictionary_entries()

    # print number of examples per label
    count = np.ones(7)
    for label in labels:
        count[label - 1] += 1

    print("-" * 10, "NUMBER OF EXAMPLES PER LABEL", "-" * 10)
    print(count)
    print("-" * 30)


    return (impressions, labels)
