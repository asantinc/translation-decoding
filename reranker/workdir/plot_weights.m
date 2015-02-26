load ../main/norm_one.m




figure(1)
sorted_results_un = sortrows(results_len_un,2)
plot(results_len_un(:,1), results_len_un(:,2))
xlabel('Length Feature Weight')
ylabel('BLEU score')
