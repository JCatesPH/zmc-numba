
lista = [complex()]
listb = [complex(3,4), complex(7,8)]

for i in range(1,5):
    lista.append(complex(i*5, i*5+1))

totlist = [lista, listb]

print(totlist[0])

lista.remove(lista[0])
print(totlist[0])