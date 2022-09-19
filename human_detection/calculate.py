import math


def factorial(num):
    if num == 1:
        return 1

    else:
        fact = 1
        while(num > 1):
            fact *= num
            num -= 1
        return fact


if __name__ == '__main__':

    sum = 0

    for i in range(1, 11):

        cal = int(factorial(i) * factorial(i+1))
        sum += cal

    print("The answer is:", sum)
