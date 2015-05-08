import re

def prime(n):
  return False if n < 2 else re.match(r'(11+)\1+$', '1'*n) is None

for i in range(0, 100):
  if prime(i):
    print(i)
