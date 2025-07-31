function metal = metal_processing(metal, patient)
% Rotate metal
metal_rot_prob = randi(2); % Random chcoice if rotate or not metal
if metal_rot_prob > 1
    metal_rot_angle = randi(360);
    metal = imrotate(metal, metal_rot_angle, 'bilinear', 'crop');
end

% Scale metal
metal_scale_prob = randi(2); % Random chcoice if scale or not metal
if metal_scale_prob > 1
    metal_scale = (1 - 0.6)*rand() + 0.6;
    metal = imresize(metal, metal_scale);
    metal = padarray(metal, [size(patient, 1) - size(metal, 1) size(patient, 2) - size(metal, 2)],'post');
end
end

