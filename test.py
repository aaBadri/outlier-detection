import random

random_vector = []
count0=0
count1=0
count2=0

for j in range(0, 100):
    x = random.randint(-1, 1)
    random_vector.append(x)
    if x==0:
        count0 += 1
    elif x==1:
        count1 += 1
    else :
        count2 += 1

print(count0,count1,count2)
print(random_vector)
