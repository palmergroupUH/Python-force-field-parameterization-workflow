import IO
import sys 
import itertools 

a = ( x for x in range(10,20))

b =( x for x in range(1,5)) 
c =( x for x in range(100,105)) 

a = itertools.chain(a,b) 

gg = itertools.chain(a,c) 

for i in gg:

    print (i) 




a = ( i for i in range(5000000) )

b = itertools.islice(a,1000) 

print (next(b))
