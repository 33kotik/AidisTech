# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        s = input()
        first = 0
        last = len(s) - 1
        flag = False
        chet = (len(s)) % 2
        print(first)
        print(last)
        print(chet)
        while s[first] == "a" and s[last] == "a":
            print(first)
            first += 1
            last -=1
            if (first == 1+last and chet == 1) or (first == last+2  and chet==0):
                flag=True
                print("NO")
                break
        if flag!=1:
            if s[first]=="a":
                print("a"+s)
            else:
                print(s+"a")
