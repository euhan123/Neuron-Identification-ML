folder_mat = dir("./Ground_truth/*.mat");

for i = 1:length(folder_mat)
    mat_name = folder_mat(i).name;
    map1 = zeros(900);
    mat_file = load(strcat("./Ground_truth/", mat_name));
    row = mat_file.info{1,1}.row;
    col = mat_file.info{1,1}.col;
    map1(row:row+333, col:col+333) = 1;
    save(strcat("./Ground_truth_heatmap/", mat_name), 'map1');
end

disp("done");