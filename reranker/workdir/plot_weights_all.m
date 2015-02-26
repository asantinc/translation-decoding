load ../norm/one/norm_one.m
load ../norm/optimum/norm_optimum.m
load ../unnorm/one/unnorm_one.m
load ../unnorm/optimum/unnorm_optimum.m

load ../main/norm_one_correct.m

close all;

figure(1);
norm_one = vec2mat(norm_one_correct, 2);
sorted_norm_one = sortrows(norm_one,2);
sorted_norm_one(end, :)
plot(norm_one(:,1), norm_one(:,2));
title('Normalized, One values')
xlabel('Length Feature Weight');
ylabel('BLEU score');


figure(2);
subplot(2,2,1)
norm_one = vec2mat(norm_one, 2);
sorted_norm_one = sortrows(norm_one,2);
sorted_norm_one(end, :)
plot(norm_one(:,1), norm_one(:,2));
title('Normalized, One values')
xlabel('Length Feature Weight');
ylabel('BLEU score');

subplot(2,2,2)
norm_optimum = vec2mat(norm_optimum,2);
sorted_norm_optimum = sortrows(norm_optimum,2);
sorted_norm_optimum(end, :)
plot(norm_optimum(:,1), norm_optimum(:,2));
title('Normalized, Optimum values')
xlabel('Length Feature Weight');
ylabel('BLEU score');

subplot(2,2,3)
unnorm_one = vec2mat(unnorm_one, 2);
sorted_unnorm_one = sortrows(unnorm_one,2);
sorted_unnorm_one(end, :)
plot(unnorm_one(:,1), unnorm_one(:,2));
title('Unnormalized, One values')
xlabel('Length Feature Weight');
ylabel('BLEU score');

subplot(2,2,4)
unnorm_optimum = vec2mat(unnorm_optimum,  2);
sorted_unnorm_optimum = sortrows(unnorm_optimum,2);
sorted_unnorm_optimum(end, :)
plot(unnorm_optimum(:,1), unnorm_optimum(:,2));
title('Unnormalized, Optimum values');
xlabel('Length Feature Weight');
ylabel('BLEU score');

