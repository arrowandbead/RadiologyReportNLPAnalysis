import utilFunctions

patients = utilFunctions.getExtractEvaluationInformation("../data/rawData/RMS/evaluation of reports.xlsx")
files = utilFunctions.iterateThroughFilesInFolder("../data/rawData/RMS/reports")

delimeter = "-----------------------\n"

cancer_statuses = ['a', 'b', 'c', 'd', 'e']
cancer_evidences = [1,2,3,4,5,6,7]

dataFile = open("data.txt", "w+")
labelFile = open("labels.txt", "w+")

def getImpression(currFilePath):
    inImpression = False
    impressionOutput = ""
    openedFile = open(currFilePath)
    fileLines = []
    for line in openedFile:
        fileLines.append(line)
    fileLines2= fileLines.copy()
    concatLines = "".join(fileLines)
    if "end of impression" in concatLines.lower():

        for thing in fileLines2:
            print(thing)
            if "impression" in thing.lower():
                inImpression = True
            if "end of impression" in thing.lower():
                return impressionOutput + "end of impression intercept"
            elif inImpression:
                impressionOutput += thing






    for line in fileLines2:
        if "impression" in line.lower():
            inImpression = True
        if "end of impression" in line.lower():
            print("end of impression")
            return impressionOutput + "end of impression"

        if inImpression:
            if "notification" in line.lower():
                print("notif")
                return impressionOutput + "notif"
            capitalStreak = 0
            if("impression" not in line.lower()):
                streak = ""
                for thing in line:
                    if thing.isupper():
                        streak += thing
                        capitalStreak += 1

                    else:
                        if thing ==":"  and capitalStreak > 4:
                            return impressionOutput + "Broken" + line
                        capitalStreak = 0
            print("adding to impressionOutput")
            impressionOutput += line
    return impressionOutput + "EOF"


for p in patients:
    for report in p:
        if "cancer status" not in report and "evidence of cancer" not in report:
            continue

        for f in files:
            allIn = True
            for term in report['post-MR']:
                if term not in f:
                    allIn = False
            if allIn:
                currFilePath = "../data/rawData/RMS/reports/" + f
                impression = getImpression(currFilePath)
                canc_ev = "-" if "evidence of cancer" not in report else report["evidence of cancer"]
                canc_stat = '-' if "cancer status" not in report else report["cancer status"]
                labels = canc_ev + ", " + canc_stat + '\n'

                assert(canc_ev!=canc_stat)
                dataFile.write(impression)
                dataFile.write(currFilePath + '\n')
                dataFile.write(delimeter)
                labelFile.write(labels)
                labelFile.write(delimeter)
