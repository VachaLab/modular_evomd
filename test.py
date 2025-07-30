from sequence import Sequence

seq = Sequence('FRRLQKYNSIAYKTLWKIQSFW')
one, other = seq.get_faces()
positions = seq.get_positions()

print(one)
print(other)
print(positions)