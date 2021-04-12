from time import sleep
from datetime import datetime
from sys import argv

def set_params():
    argc = len(argv) - 1
    if argc > 0:
        for index, arg in enumerate(argv[1:]):
            if arg == "-K":
                if argc > index+1:
                    global K
                    K = int(argv[2+index])
                else:
                    print("Please give a value after -K!")


print("argv:", argv)
K = 10000
set_params()
ti = datetime.now()


for i in range(K):
    sleep(0.1)

tf = datetime.now()

print(f"Done! with K = {K}")
print(f"Time Elapsed = {tf - ti}")
print(f"Estimated Time = {K * 0.1}")
