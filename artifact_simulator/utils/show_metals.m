function [] = show_metals(CT_samples_bwMetal)
for i = 1:size(CT_samples_bwMetal, 3)
    imshow(CT_samples_bwMetal(:,:,i))
    title(i)
    waitforbuttonpress  
end
end

