from sequence import Sequence

seq =   Sequence('FRRLQKYNSIAYKTLWKIQSFW')
other = Sequence('TNNQQALHQMRNNYNRLNNIVQ')

print(('{}'.format(seq[:13])))
print(len(seq), len(other))
print(seq[:3] + other[3:6] + seq[6:])