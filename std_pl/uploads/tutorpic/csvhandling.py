import csv
fr=open("fruitfile.csv")
csvr=csv.reader(fr)
header=[]
header=next(csvr)
for hd in header:
    print(hd,end="-")

rows=[]
print()
for r in csvr:
    rows.append(r)

for rw in rows:
    print(rw[0],rw[1],rw[2])
    

