% =========================================================================================================
%       C Q UC Q UC Q UC Q U          C Q UC Q UC Q U              C Q U          C Q U
% C Q U               C Q U     C Q U               C Q U          C Q U          C Q U
% C Q U                         C Q U               C Q U          C Q U          C Q U
% C Q U                         C Q U               C Q U          C Q U          C Q U
% C Q U                         C Q U               C Q U          C Q U          C Q U
% C Q U                         C Q UC Q UC Q U     C Q U          C Q U          C Q U
% C Q U               C Q U     C Q U          C Q UC Q U          C Q U          C Q U
%      C Q UC Q UC Q U               C Q UC Q UC Q U                    C Q UC Q U
%                                              C Q UC Q U
%
%     (C) Copyright Chongqing University All Rights Reserved.
%
%     This program and corresponding materials are protected by software
%     copyright and patents.
%
%     Corresponding author：Ran Liu
%     Address: College of Computer Science, Chongqing University, 400044, Chongqing, P.R.China
%     Phone: +86 136 5835 8706
%     Fax: +86 23 65111874
%     Email: ran.liu_cqu@qq.com
%
%     Filename         : main.m
%     Description      :
%   ------------------------------------------------------------------------
%       Revision   |     DATA     |   Authors                     |   Changes
%   ------------------------------------------------------------------------
%         1.00     |  2014-10-28  |   Donghua Cao                 |   Original
%         1.01     |  2014-10-28  |   Ran Liu                     |   The code style has been changed
%         1.02     |  2018-09-15  |   Ran Liu                     |   Modules of median filtering and big hole dilation have been added
%         1.03     |  2019-12-11  |   ZYT                         |   Module of threed_image_warping has been modified for improved DIBR method
%   ------------------------------------------------------------------------
% =========================================================================================================
clc
clear
% 全局变量
% initialise parameters
calib_params_balloons; % read the calibration parameters of “balloons” sequence
data_path = 'E:\Desktop\research\dibr\paper\nowtest\dataset\balloons\';
mask_folder_root_path = 'E:\Desktop\research\dibr\paper\nowtest\dataset\balloons\instance\balloons5\';

save_path = 'E:\Desktop\research\dibr\paper\nowtest\dataset\balloons\dibr';

%% 3D image warping for 300 frames
hole_pers = zeros(300,2);
warp_time = zeros(300,1);
for idx = 0 : 99
    t1=clock;
    %% input images and maps
     if (idx < 10)
        FileName = strcat(data_path, 'images\balloons5\color-cam5-f00', int2str(idx), '.png');
        IR = imread(FileName);
        
        % 读取IR对应的深度图
        FileName = strcat(data_path, 'depth\balloons5\depth-cam5-f00', int2str(idx), '.png');
        D = imread(FileName);
        
         % 读取对应的instance图
        frame_name = strcat('color-cam5-f00', int2str(idx));
    else
        FileName = strcat(data_path, 'images\balloons5\color-cam5-f0', int2str(idx), '.png');
        IR = imread(FileName);
        
        % 读取IR对应的深度图
        FileName = strcat(data_path, 'depth\balloons5\depth-cam5-f0', int2str(idx), '.png');
        D = imread(FileName);
        
         % 读取对应的instance图
        frame_name = strcat('color-cam5-f0', int2str(idx));
     end


    individual_masks_path = fullfile(mask_folder_root_path, frame_name, '/');
    
    if ~exist(individual_masks_path, 'dir')
        error('找不到实例掩码文件夹: %s', individual_masks_path);
    end

   
    mask_files_jpg = dir(fullfile(individual_masks_path, '*.jp*g'));
    mask_files_png = dir(fullfile(individual_masks_path, '*.png'));
    all_mask_files = [mask_files_jpg; mask_files_png]; % 合并所有找到的图片文件

   
    filtered_mask_names = {}; % 使用元胞数组(cell array)来存储筛选后的文件名
    for i = 1:length(all_mask_files)
        file_name = all_mask_files(i).name;
        % 如果文件名【不】是以 '_0.jpg' 或 '_0.png' 结尾, 就保留
        if ~endsWith(file_name, ['_0.jpg']) && ~endsWith(file_name, ['_0.png'])
            filtered_mask_names{end+1, 1} = file_name;
        end
    end
    % 对筛选后的文件名进行排序, 确保处理顺序是 _1, _2, ...
    sorted_mask_names = sort(filtered_mask_names); 
    
   
    [hi, wi, ~] = size(IR);
    instance_map = uint8(zeros(hi, wi)); 
    
   
    for k = 1:length(sorted_mask_names)
        % 定义当前实例的ID (从1开始递增)
        instance_id = uint8(k); 
        
        % 读取单个实例的二值掩码文件
        single_mask_filename = fullfile(individual_masks_path, sorted_mask_names{k});
        single_mask_img = imread(single_mask_filename);
         
        % 为确保健壮性, 将读入的掩码转为逻辑型 (logical)
        if size(single_mask_img, 3) > 1
            single_mask_img = single_mask_img(:,:,1); 
        end
        is_instance_region = (single_mask_img > 0); 
       
        % 使用逻辑索引, 将当前实例ID赋值给画布上的对应区域
        instance_map(is_instance_region) = instance_id;
    end


%     % 读取摄像机5捕获的图像
%     FileName = strcat(data_path, 'images\balloons5\color-cam5-f', int2str(idx), '.png');
%     IR = imread(FileName);
%     
%     % 读取IR对应的深度图
%     FileName = strcat(data_path, 'depth\balloons5\depth-cam5-f', int2str(idx), '.png');
%     D = imread(FileName);
%     
%     % 读取对应的instance图
%     FileName = strcat(mask_path, 'color-cam5-f', int2str(idx), '_mask.png');
%     instance_map = imread(FileName);
    
    hi = size(IR, 1);
    wi = size(IR, 2);
    
    %% 3D image warping
    % M: generated non-hole matrix, double
    %       -1: current point is a hole-point
    %       [0, 255]: depth value of the current point
%     [Ides, M] = threed_image_warping_left(IR, D, MinZ, MaxZ, m1_ProjMatrix, m5_RT, m5_ProjMatrix);
      %计算宽度大于3的最小深度越变TD
%     if (idx == 0)
%        T_D = TD(IR, D, MinZ, MaxZ,  m1_ProjMatrix, m5_RT, m5_ProjMatrix);
%        fprintf('T_D  = %d', T_D );
%     end  
    
    
    [Ides, M, Ides_bk, M_bk, fg_map] = threed_image_warping_left_wmask(instance_map, IR, D, MinZ, MaxZ,  m1_ProjMatrix, m5_RT, m5_ProjMatrix);% Ides: generated destination image, uint8; M: generated non-hole matrix, double
    
    
    % 统计hole size
    hole_nums = M==-1;
    hole_nums = sum(sum(hole_nums));
    hole_per = hole_nums/(hi*wi);
    hole_pers(idx + 1,1) = hole_per;
    % for test
    % 输出三维图像变换结果
    %         FileName_Ides = strcat('Ides\Ides_TIW_', int2str(idx), '.bmp');
    %         imwrite(Ides, FileName_Ides);
    %         FileName_M = strcat('M\M_TIW_', int2str(idx), '.txt');
    %         save(FileName_M, 'M',  '-ASCII');
    %
%             figure(3);
%             imshow(Ides);
%             figure(4);
%             fig_m = M;
%             fig_m(fig_m==-1)=0;
%             fig_m = uint8(fig_m);
%             imshow(fig_m);
    
    %% non-hole matrix filtering
    % % for test;
    % FileName_Ides = strcat('Ides\Ides_TIW_', int2str(idx), '.bmp');
    % Ides = imread(FileName_Ides);
    % FileName_M = strcat('M\M_TIW_', int2str(idx), '.txt');
    % M = load(FileName_M);
    %
%     [Ides, M] = median_filtering(Ides, M, wi, hi);
    %
    % % save Ides and M，在当前目录下
    % FileName_Ides = strcat('Ides\Ides_MF_', int2str(idx), '.bmp');
    % imwrite(Ides, FileName_Ides);
    % FileName_M = strcat('M\M_MF_', int2str(idx), '.txt');
    % save(FileName_M, 'M',  '-ASCII');
    
    %        for test
    %         figure(3);
    %         imshow(Ides);
    %         figure(4);
    %         imshow(M);
    
     %%  big_hole_dilation
    %     for test;
%     FileName_Ides = strcat('Ides\Ides_TIW_', int2str(idx), '.bmp');
%     Ides = imread(FileName_Ides);
%     FileName_M = strcat('M\M_TIW_', int2str(idx), '.txt');
%     M = load(FileName_M);
%     figure(1);
%     imshow(Ides);
    
%     a = Ides; % for test
    
%     figure(2);
%     imshow(M);
    
    th_big_hole = 3; % threshold for big hole detection. if the number of hole-points in a hole is greater than th_big_hole, the hole is labeled as a “big hole”
    sharp_th = 4; % threshold for sharp transition
    n_dilation = 3; % the number of points to be dilated
    rend_order = 0; % flag of the rending order of the reference image. The destination image positioned at camera 4 is rendered from camera 5 (camera 5 ? camera 4), therefore the destination image is right view.
    [Ides, M] = big_hole_dilation(Ides, M, th_big_hole, sharp_th, n_dilation, rend_order, wi, hi);
    [Ides_bk, M_bk] = big_hole_dilation(Ides_bk, M_bk, th_big_hole, sharp_th, n_dilation, rend_order, wi, hi);
    [fg_map, M] = big_hole_dilation(fg_map, M, th_big_hole, sharp_th, n_dilation, rend_order, wi, hi);
%     [Ides_map, M_map] = big_hole_dilation(Ides_map, M_map, th_big_hole, sharp_th, n_dilation, rend_order, wi, hi);
    % 统计hole size
    hole_nums = M==-1;
    hole_nums = sum(sum(hole_nums));
    hole_per = hole_nums/(hi*wi);
    hole_pers(idx+1,2) = hole_per;

    %% 保存目标图像深度信息
     D = M;
     
          %保存深度信息，可不用
%      FileName_D = strcat('G:\thesis_experiment_rxw\datasets\Balloons\improved_dibr\D_balloons\Is_balloons_D', int2str(idx), '.txt');
%      save(FileName_D, 'D', '-ASCII');
     
     D(D==-1)=255;  %如果想要深度图中空洞区域为黑色，只需要将值设为0即可
     D = uint8(D);
     
%      %保存空洞值设为255的深度图
%      FileName_dep = strcat('../datasets/ballet/improved_dibr/Depthmap_ballet/Is_ballet_depth_', int2str(idx), '.png');
%      imwrite(D, FileName_dep);
%     
     %% 保存目标图和只含背景信息的目标图
    %保存目标图像,空洞部分是黑色的
%      FileName_Ides = strcat(save_path, 'black_hole/D_ballet/Is_ballet_des', int2str(idx), '.png');
%      imwrite(Ides, FileName_Ides);
     %保存只含背景的目标图像，空洞部分是黑色的
%      FileName_Ides_bk = strcat(save_path, 'black_hole/D_ballet_bk/Is_ballet_bk_des', int2str(idx), '.png');
%      imwrite(Ides_bk, FileName_Ides_bk);
     
    %% 保存用于cnn的图像。黑白翻转:翻转后，0代表非空像素点，1代表空洞像素点
    M(M ~= -1) = 1;
    M(M == -1) = 0;
    for i = 1 : hi
        ind = find(M(i,:) == 0);
        Ides(i, ind, :) = 255;
    end
    M =~ M;
    
    M_bk(M_bk ~= -1) = 1;
    M_bk(M_bk == -1) = 0;
    for i = 1 : hi
        ind = find(M_bk(i,:) == 0);
        Ides_bk(i, ind, :) = 255;
    end
    M_bk =~ M_bk;
    
    %对fg_map 进行闭操作
    %se = strel('disk', 15);
    %fg_map = imclose(fg_map, se);

    
 % 1. 定义Dilation的结构元素 (您可以根据需要调整 'disk' 的大小)
    se = strel('disk', 3); % 例如, 使用一个半径为5像素的圆形结构元素
    
    % 2. 初始化一个空的画布, 用于存放最终合并的掩码
    final_combined_mask = false(hi, wi);
    
    % 3. 获取图中所有 warping 后的实例ID
    unique_ids = unique(fg_map);
    unique_ids(unique_ids == 0) = []; % 移除背景ID 0
    
    % 4. 遍历每一个实例
    for k = 1:length(unique_ids)
        instance_id = unique_ids(k);
        
        % a. 提取当前实例原始的、未膨胀的二值掩码
        original_instance_mask = (fg_map == instance_id);
        
        % b. 对该实例掩码进行膨胀操作
        dilated_instance_mask = imdilate(original_instance_mask, se);
        
        % c. 将膨胀后的掩码与空洞掩码 M 取交集
        intersection_mask = dilated_instance_mask & M;
        
        % d. 判断是否存在交集
        if any(intersection_mask(:))
            % e. 如果存在交集, 就将该实例【原始的】掩码合并到最终的画布上
        final_combined_mask = final_combined_mask | original_instance_mask;
        
        end
    end
    
    % 5. 将最终的逻辑掩码转换为 uint8 格式 (0 和 255)
    final_mask_uint8 = uint8(final_combined_mask) * 255;



    t2=clock;
    warp_time(idx + 1,1)=etime(t2,t1);
    
% 生成格式化的三位数索引字符串
idx_str = sprintf('%03d', idx);


save_folder_Ides = fullfile(save_path, 'Ides_Balloons');
if ~exist(save_folder_Ides, 'dir')
    mkdir(save_folder_Ides);
end

FileName_Ides = fullfile(save_folder_Ides, ['color-cam1-f', idx_str, '.png']);
imwrite(Ides, FileName_Ides);

save_folder_M = fullfile(save_path, 'M_Balloons');
if ~exist(save_folder_M, 'dir')
    mkdir(save_folder_M);
end
% 保存Mask
FileName_M = fullfile(save_folder_M, ['color-cam1-f', idx_str, '.png']);
imwrite(M, FileName_M);



save_folder_Ides_bk = fullfile(save_path, 'Ides_Balloons_bk');
if ~exist(save_folder_Ides_bk, 'dir')
    mkdir(save_folder_Ides_bk);
end


FileName_Ides_bk = fullfile(save_folder_Ides_bk, ['color-cam1-f', idx_str, '.png']);
imwrite(Ides_bk, FileName_Ides_bk);

save_folder_M_bk = fullfile(save_path, 'M_Balloons_bk');
if ~exist(save_folder_M_bk, 'dir')
    mkdir(save_folder_M_bk);
end


FileName_M_bk = fullfile(save_folder_M_bk, ['color-cam1-f', idx_str, '.png']);
imwrite(M_bk, FileName_M_bk);




save_folder_fg_map = fullfile(save_path, 'fg_instance');
if ~exist(save_folder_fg_map, 'dir')
    mkdir(save_folder_fg_map);
end
% 保存前景物体mask
FileName_fg_map = fullfile(save_folder_fg_map, ['color-cam1-f', idx_str, '.png']);
imwrite(final_mask_uint8, FileName_fg_map);
    
    
end
% save('ballet_hole_pers.mat','hole_pers');
save('Balloons_warp_time.mat','warp_time');