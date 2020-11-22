from openpyxl import load_workbook
import os

def iterateThroughFilesInFolder(folderPath):
    return os.listdir(folderPath)

def getClinicalInformationDataMap(clinicalInfoPath):
    workbook = load_workbook(clinicalInfoPath)
    sheet1 = workbook.get_sheet_by_name("clinical")
    i = 1
    for row in sheet1.iter_rows():
        i+=1
        print([cell.value for cell in row])
        if i == 10:
            exit()
