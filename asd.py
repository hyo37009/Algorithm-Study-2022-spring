def inpnode():
    num = int(input('정점 수 : '))
    nodelist = [None]
    for i in range(num):
        now = list(map(int, input(f'{i + 1}과 연결된 노드 : ').split(' ')))
        if type(now) != int:
            nodelist.append(None)
        nodelist.append(now)

    return nodelist

def doConneted(nodelist, r, connet=None):
    if connet is None:
        connet = []
    if nodelist is None:
        return False
    if r == 0:
        return True
    for i in nodelist:
        connet.append(i)
        doConneted(nodelist[i], r - 1)
        connet.pop()

nodelist = inpnode()
print(doConneted(nodelist, 2))