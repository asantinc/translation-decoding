load 'q3/morph-country.csv'
load 'q3/morph-countries.csv'
for i = 1:1
    figure; bar(morph_countries(i,:));
    axis([0,8750,-inf,inf]);
    figure; bar(morph_country(i,:));
    axis([0,8750,-inf,inf]);
    
end