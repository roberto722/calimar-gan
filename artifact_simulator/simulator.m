clc, clear, close all; 
addpath('.\src');
addpath('.\utils');
multiWaitbar('CloseAll');

% Output folders
folder_GT  = "processed_dataset/gt/"; % No artifacts images (Ground Truth) for backup
folder_sim_art = "processed_dataset/art/"; % Artifacts images and gt.mat

% Source masks folder
folder_masks = "masks\";

% Settings
random_centroid = false;
number_of_masks = 6; % Number of masks to generate (= 25 in the official implementation)

% Utils
if not(isfolder(folder_GT))
    mkdir(folder_GT)
end
if not(isfolder(folder_sim_art))
    mkdir(folder_sim_art)
end

%% Load images & masks
listing_SW_cases = struct2table(dir("raw_dataset\**\*.nii.gz"));
SW_filelist = listing_SW_cases.name;

% Loading available metal masks list
listing_masks = struct2table(dir("masks\*.png"));
masks_filelist = listing_masks.name;
num_masks = numel(masks_filelist);
for m = 1:num_masks
    masks{m} = imread(string(listing_masks.folder(m)) + '\' + string(listing_masks.name(m)));
    multiWaitbar('Loading masks', m/num_masks, 'CanCancel', 'off');
end
multiWaitbar('CloseAll');

% Min and Max slices to avoid selecting head and pelvis slices
slices_data = readtable('raw_dataset\UWSpineCT-meta-data.csv', 'VariableNamingRule', 'preserve');
disp("Successfully loaded cases index and metal masks")

%% Create spacing info csv
for i = 1:numel(SW_filelist)
    spacing_info(i, 1) = str2double(erase(SW_filelist{i}, '.nii.gz'));
    patient_case_folder = strcat(string(listing_SW_cases.folder(i)), '\', string(SW_filelist{i}));
    patient_info = niftiinfo(patient_case_folder);
    spacing_info(i, 2) = patient_info.PixelDimensions(1);
    multiWaitbar('Loading spacing info', i/numel(SW_filelist), 'CanCancel', 'off');
end
save("spacing_info.mat", "spacing_info")
multiWaitbar('CloseAll');

%% Start Processing cases
% Iterate over cases
for i = 1:size(listing_SW_cases, 1)
    fprintf("Case %d of %d ... \n", i, size(SW_filelist, 1))
    case_match = find(slices_data{:,2} == str2double(SW_filelist{i}(1:7)));

    multiWaitbar('Processing cases', i/size(listing_SW_cases, 1), 'CanCancel', 'off');
    if isempty(case_match) == 0
        patient_case_folder = strcat(string(listing_SW_cases.folder(i)), '\', string(SW_filelist{i}));
        % Reading Case Volume
        patient = niftiread(patient_case_folder);
        % Reading Case Info
        patient_info = niftiinfo(patient_case_folder);

        SOD = (patient_info.raw.pixdim(2) / 5) * 256; % Source to Origin Distance
        %SOD = sqrt(size(patient,1)^2 + size(patient,2)^2) + 50;
        pixel_size = patient_info.raw.pixdim(4)/100;

        % Set config
        config = set_config_for_artifact_simulation(pixel_size, SOD);
        disp('Set config');

        % Phantom calibration
        phantom = create_phantom(512, 512, 200, config.mu_water);
        config.correction_coeff = water_correction(phantom, config);
        disp('Phantom calibration');

        idx_start = slices_data{case_match, "Min Slice"};
        idx_end = slices_data{case_match, "Max Slice"};
        for j = idx_start:idx_end
            slice = patient(:, :, j);
            multiWaitbar('Processing slices', (j-idx_start)/(idx_end-idx_start), 'CanCancel', 'off');
            if max(slice,[],'all') < 2500
                tic
                % Preprocess
                slice(slice<-1000) = -1000; % erase the boundary
                image = hu2mu(double(slice), config.mu_water, config.mu_air);

                % Metal centering on bones
                % Find bones
                slice_bones = logical(slice);
                slice_bones(slice>=500) = 1;
                slice_bones(slice<500) = 0;
                slice_bones = imclose(slice_bones, strel('disk', 5));

                % Find bones centroids
                image_props = regionprops("table", slice_bones, 'centroid', 'Area');
                image_props = sortrows(image_props, "Area", "descend");

                % Select only centroids with 400 pixel or more
                good_centroids_idxs = image_props.Area > 400;

                if sum(good_centroids_idxs) ~= 0
                    bones_centroids = cat(1, image_props.Centroid(good_centroids_idxs,:));
                    if random_centroid == true
                        % Randomly pick one centroid from the available ones
                        selected_centroid_idx = randi(size(bones_centroids, 1));
                        selected_centroid = bones_centroids(selected_centroid_idx, :);
                    else
                        % Pick the biggest centroid from the available ones
                        selected_centroid = bones_centroids(1, :);
                    end

                    % Select Random Metals from available list
                    metal_idxs = randperm(size(listing_masks,1), number_of_masks);
                    for mtl_idx = 1 : number_of_masks
                        metal = masks{metal_idxs(mtl_idx)};
                        %imshow(metal);
                        
                        metal = logical(metal_processing(metal, patient)); % Rotate and scale metal
                        
                        % Find metal centroids
                        metal_props = regionprops("table", metal, 'centroid', 'Area');
                        metal_centroids = cat(1, metal_props.Centroid);
                        metal_centroids_mean = mean(metal_centroids, 1); % Find one single centroid
                        
                        % Couple centroids
                        metal_translated = imtranslate(metal, [selected_centroid(1) - metal_centroids_mean(1), selected_centroid(2) - metal_centroids_mean(2)]);

                        % Deleting metal outside CT acquisition circle
                        m_circle_strel = strel('disk', 256, 0);
                        m_circle = m_circle_strel.Neighborhood(1:size(slice, 1), 1:size(slice, 2));
                        % imshow(m_circle_strel.Neighborhood)
                        % imshow(metal_translated)
                        metal_translated(~m_circle) = 0;

                        metal_final = single(metal_translated);

                        % metal_fig = figure;
                        % imshow(metal_translated)
                        % circle = images.roi.Circle(gca, "Center", [size(slice, 1)/2 size(slice, 2)/2], "Radius", (size(slice, 1)-1)/2);
                        % circle_mask = createMask(circle);
                        % close(metal_fig)
                        % circle_mask = imcomplement(circle_mask);
                        % 
                        % metal_final = regionfill(single(metal_translated), circle_mask);
                           
                        % Metal Artifact Simulation
                        sim = metal_artifact_simulation(image, metal_final, config);

                        % Convert results from mu to HU
                        sim_hu = mu2hu(sim, config.mu_water, config.mu_air);

                        image_w_art = linear_ct_window(sim_hu);

                        % Saving image with artifact in HU and 0-255
                        path2 = sprintf("%s%s/%s",folder_sim_art, SW_filelist{i}(1:7), int2str(j));
                        check_path(path2)
                        sim_hu_int = int16(sim_hu);
                        save(path2 + "\" + mtl_idx + "_HU.mat", "sim_hu_int")
                        save(path2 + "\" + mtl_idx + ".mat", "image_w_art")
                        save(path2 + "\" + mtl_idx + "_mask.mat", "metal_translated")

                        % Thumbnails - File name: patient id + metal mask
                        imwrite(imresize(image_w_art, [256 256]), folder_sim_art + SW_filelist{i}(1:7) + "_" + j + "_" + mtl_idx + ".png", 'fmt', "png")
                        
                        %%%% FOR DEBUGGING
                        % figure
                        % subplot(1, 3, 1); imshow(image)
                        % subplot(1, 3, 2); imshow(metal_final)
                        % hold on 
                        % plot(bones_centroids(:, 1), bones_centroids(:, 2), 'g*')
                        % plot(selected_centroid(1), selected_centroid(2), 'r*')
                        % hold off
                        % subplot(1, 3, 3); imshow(linear_ct_window(sim_hu))                     
                    end    

                    % Saving ground truth in HU 
                    % File name: gt.mat
                    save(path2 + "\gt.mat", "slice")  
                    fprintf("Processed slice %d of %d ... Time elapsed: %.2f s \n", j, size(patient, 3), toc)
                    imwrite(imresize(linear_ct_window(slice), [256 256]), folder_GT + SW_filelist{i}(1:7) + "_" + j + "_gt.png")
                else
                    disp("Skippink slice: no good bones found :(")
                end
            end 
        end
    else
        disp("Case not containing CSV data... skipping...")
    end
end

%% Functions
function check_path(folder)
    if not(isfolder(folder))
        mkdir(folder)
    end
end
