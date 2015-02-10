
dir = './pic/';
index = zeros(Database.cnt,1);
for i = 1 : Database.cnt
    Showimg(Database.data{i},dir);
    disp(Database.data{i}.filename);
    pause();
end