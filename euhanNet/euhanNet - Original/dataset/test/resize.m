folder = dir('Ground_truth/*.jpg');

for i = 1:100
    name = folder(i).name;
    
    img = imread(fullfile('Ground_truth',name));
    
    re = imresize(img, [80, 80]);
    re = imresize(re, [320 320]);
   
    imwrite(re, strcat('./Images/',name));
end