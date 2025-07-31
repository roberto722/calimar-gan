function rescaled_img = linear_ct_window(ct_img)
    ct_img = double(ct_img);
    ct_window = [3600, 700];
    img_min = (ct_window(2) - ct_window(1))/2; %-1450
    img_max = (ct_window(1) + ct_window(2))/2; %+2100
    ct_img(ct_img < img_min) = img_min;
    ct_img(ct_img > img_max) = img_max;
    

    % Rescaling
    ct_img(ct_img <= 1000) = 0.0004286*ct_img(ct_img <= 1000) + 0.471429;
    ct_img(ct_img > 1000) = 0.0000909*ct_img(ct_img > 1000) + 0.8090909;

    rescaled_img = uint8(ct_img * 255);
end
