"""
0,1,1,2,3,5,8...
"""
a=0
b=1
i=0
print("0,1,",end="")
while i<=10:
    c=a+b
    a=b
    b=c
    print(c,end=",")
    i=i+1
    
    

    
