folder_1x1 = dir("./Images_1x1/*.png");

for i = 1:length(folder_1x1)
    name_1x1 = folder_1x1(i).name;
    image_1x1 = imread(strcat("./Images_1x1/", name_1x1));
    for j = 1:10
        imwrite(image_1x1, strcat("./Images_1x1/", name_1x1(1:end-4), string(j), ".png"));
    end
end
