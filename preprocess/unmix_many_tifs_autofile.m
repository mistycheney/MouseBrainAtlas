image_directory = [pwd,'\']; % get images from working directory
%image_directory = 'C:\Users\DKLab\Documents\MATLAB'; % get images from somewhere else

color1_name = 'c1_ORG.tif';
color2_name = 'c2_ORG.tif';

counter = 1;
c1_list = {};
c2_list = {};

allcontents = dir;
numContents = length(allcontents);
for icontent = 1:numContents
    name = allcontents(icontent).name;
    k = strfind(name, color1_name);
    if ~isempty(k)
        sectionname = name(1:k-1);
        c1_list{counter} = [sectionname,color1_name];
        c2_list{counter} = [sectionname,color2_name];
        counter = counter+1;
    end
end
                    
bg_list = c1_list; %bg is blue, background
image_list = c2_list;


if length(image_list) ~= length(bg_list)
    error('uh oh ... length of image and background image lists not same');
end

for iimg = 1:length(image_list)
    disp(['Processing image #',num2str(iimg)]);
    unmix_tif_GB_separate_input_channels([image_directory,image_list{iimg}],[image_directory,bg_list{iimg}]);
end


disp('Done!');