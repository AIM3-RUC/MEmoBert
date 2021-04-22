x = 'abc'
y = 'abcd'
flag=False
for i in range(len(y)-len(x)+1):
  for j in range(len(x)):
    if x[j] == y[i+j]:
      flag = True
    else:
      flag = False
      break
  if flag:
    break
print('is {} as a subset of {}? {}'.format(x, y, flag))