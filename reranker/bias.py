import normalize
import pdb

unnorm = open('train/unnorm/len.out', 'w')
in_f = open('train/100best_clean.out', 'r')

pdb.set_trace()
for line in in_f:
    print len(line)
    unnorm.write(str(len(line.strip()))+'\n')
    
normalize.normalize('train/unnorm/len.out', 'train/norm/len.out')
