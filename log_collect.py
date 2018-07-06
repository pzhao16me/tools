
import os 
def get_logs_file(rootDir):
    global log_path
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if path.endswith(".log"):
            #print path
            log_path.append(path)

        if os.path.isdir(path): 
            get_logs_file(path)
    return log_path


# test
rootDir = os.getcwd()


log_path = []
log_path = get_logs_file(rootDir)

print "th len of log_path is :", len(log_path)

for path in log_path:
    print path
    file = open(path, "r")
    file_content = file.readlines()
    with open("tmp.log", "a") as tmp:
        tmp.writelines(file_content)

