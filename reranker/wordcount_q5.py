from rerankfun_q5 import *
from compute_bleu_function import *
import variance as var




lm = 1
tm = 1
lex = 1
normalize = True
output = open('norm/one/norm_one.m', 'w')
for len_w in range(-40, 120):
    f = rerank(var.mean_feats, var.std_feats, lex,tm,lm,len_w,'norm/one/len'+str(len_w)+'One', normalize)
    output.write(str(len_w)+','+str(compute_bleu(f))+';')


lm = 32
tm = 4
lex = 16
output = open('norm/optimum/norm_optimum.m', 'w')
for len_w in range(-40, 120):
    f = rerank(var.mean_feats, var.std_feats, lex,tm,lm,len_w,'norm/optimum/len'+str(len_w)+'optimum', normalize)
    output.write(str(len_w)+','+str(compute_bleu(f))+';')


lm = 1
tm = 1
lex = 1
normalize = False
output = open('unnorm/one/unnorm_one.m', 'w')
for len_w in range(-40, 120):
    f = rerank(var.mean_feats, var.std_feats, lex,tm,lm,len_w,'unnorm/one/len'+str(len_w)+'optimum', normalize)
    output.write(str(len_w)+','+str(compute_bleu(f))+';')


lm = 32
tm = 4
lex = 16
output = open('unnorm/optimum/unnorm_optimum.m', 'w')
for len_w in range(-40, 120):
    f = rerank(var.mean_feats, var.std_feats, lex,tm,lm,len_w,'unnorm/optimum/len'+str(len_w)+'optimum', normalize)
    output.write(str(len_w)+','+str(compute_bleu(f))+';')



