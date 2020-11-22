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
