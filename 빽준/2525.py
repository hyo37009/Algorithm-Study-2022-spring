a, b = map(int, input().split(' '))
b += int(input())
if b >= 60:
    a += b // 60
    b = b%60
a %= 24
print(a, end=' ')
print(b)
