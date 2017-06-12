fid = fopen('./picin_3.bin');
[param_data,count_param_conv1] = fread(fid,'float32');
fclose(fid);
param_data = reshape(param_data,[28,28]);
imshow(param_data');
