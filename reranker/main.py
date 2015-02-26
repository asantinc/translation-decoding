from rerankfun_q5 import *
from compute_bleu_function import *
import variance as var


lm = 1
tm = 1
lex = 1
normalize = True

output = open('main/norm_one.m', 'w')
for len_w in range(-40, 120):
    f = rerank(var.mean_feats, var.std_feats, lex,tm,lm,len_w,'main/norm_one_len'+str(len_w), normalize)
    output.write(str(len_w)+','+str(compute_bleu(f))+';'+'\n')
output.close()



