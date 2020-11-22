from openpyxl import load_workbook
import os

def iterateThroughFilesInFolder(folderPath):
    return os.listdir(folderPath)

def getFieldsFromXlsx(filePath, fields):
    workbook = load_workbook(filePath)
    sheet1 = workbook.get_sheet_by_name("clinical")
    iter = sheet1.iter_rows()
    row0 = [cell.value for cell in next(iter)]
    fieldToIndexMap = {}
    for index, thing in enumerate(row0):
        fieldToIndexMap[thing] = index
    outPutList = []
    for row in iter:
        newDict = {}
        rowAsList = [cell.value for cell in row]
        for thing in fields:
            newDict[thing] = rowAsList[fieldToIndexMap[thing]]
        outPutList.append(newDict)
    return outPutList


def getExtractEvaluationInformation(filePath):
    workbook = load_workbook(filePath)
    sheet1 = workbook.get_sheet_by_name("sheet1")
    iter = sheet1.iter_rows()
    row0 = [cell.value for cell in next(iter)]

    indexToFieldMap = {}
    for index, thing in enumerate(row0):
        indexToFieldMap[index] = thing
    indexToFieldMap[2] = "time"

    outPutList = []
    for row in iter:
        patientReportList = []
        outPutList.append(patientReportList)
        rowAsList = list([cell.value for cell in row])
        currReportDict = None
        for index,thing in enumerate(rowAsList):
            if(index == 0 or thing == None):
                continue
            if "post-MR" in indexToFieldMap[index]:
                if currReportDict != None:
                    if "cancer status" not in currReportDict and "evidence of cancer" not in currReportDict:
                        patientReportList = patientReportList[:-1]
                currReportDict = {}
                patientReportList.append(currReportDict)
                currReportDict["post-MR"] = str(thing).strip(" ").split(',')
            else:

                if thing != "" and "#" not in str(thing):
                    currReportDict[indexToFieldMap[index]] = str(thing)
        if currReportDict != None:
            if "cancer status" not in currReportDict and "evidence of cancer" not in currReportDict:
                patientReportList = patientReportList[:-1]
    return outPutList
