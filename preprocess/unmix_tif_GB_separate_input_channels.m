function imgG_demix_uint = unmix_tif_GB_separate_input_channels(input_channel_file,bg_channel_file);
% input - single channel tif images for input channel (img), background channels (bg)
% output- single channel tif image with estimated bg channel demixed

% read in image file
imgG = imread(input_channel_file,'tif');
imgB = imread(bg_channel_file,'tif');

% squeeze pixel values into one-dimensional arrays for analysis
imgG_list = reshape(imgG,prod(size(imgG)),1);
imgB_list = reshape(imgB,prod(size(imgB)),1);


%% minimum value fit to subtract offset
nbins = 10;
B_bins = linspace(double(min(imgB_list)),double(max(imgB_list)),nbins); % create evenly spaced bins of bg pixel vals
B_bins_inds = discretize(imgB_list,B_bins); % place bg pixels in bins

% find minimum image pixel per bg pixels bin
G_min_per_B_bin = []; B_full_bins = [];
for i=1:max(B_bins_inds)
    % find minimum G value in each bin of B pixels
    minG = min(imgG_list(find(B_bins_inds==i)));
    if ~isempty(minG)
        B_full_bins = [B_full_bins ; B_bins(i)];
        G_min_per_B_bin = [G_min_per_B_bin ; minG ];
    end
end
% linear regression on minimum G values per bin
B = double([ones(size(B_full_bins)) B_full_bins])\double(G_min_per_B_bin);

offset = B(1);
scale_factor = B(2);

% only subtract offset if <0 (conservative)
if offset>0;
    offset = 0;
end

% print offset and scale factor
offset
scale_factor

% estimate bleedthrough 
GB_bleed_est = offset+scale_factor*double(imgB);

% subtract estimated bleedthrough
imgG_demix = double(imgG) - GB_bleed_est;

%% plot for debugging
% set doplot = 1 to plot

% doplot = 1
% 
% if doplot == 1
%     figure;
%     hold on;
%     plot(imgB_list,imgG_list,'g.');
%     plot(B_full_bins,G_min_per_B_bin,'b.');
%     plot(B_full_bins,B(1)+B(2)*B_full_bins,'r');
%     xlabel('background pixels');
%     ylabel('input pixels');
%     legend('input pixels','min value per bin','fit');
% end


%% format as image and write to file

% reformat adjusted channel to positive integer
class_img = class(imgG);
imgG_demix_uint = eval([char(class_img),'(imgG_demix)']);

% write adjusted image format
imwrite(imgG_demix_uint,[input_channel_file,'_demixed.tif'],'tif');
