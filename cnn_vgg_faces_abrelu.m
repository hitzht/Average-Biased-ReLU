%   This Research is funded by IIIT Sri City, INDIA through Institute Research Grant.
%   cnn_vgg_faces_abrelu.m returns the Average Biased ReLU Based CNN Face Descriptor for an input image.
%   The original trained VGGFace model is used and updated to the AB-ReLU based Face VGGFace descriptor by Dr. Shiv Ram Dubey, IIIT Sri City.
%   The original trained VGGFace model is available at: http://www.vlfeat.org/matconvnet/pretrained/
%   This code can be used only for the academic and research purposes and can not be used for any commercial purposes.
%   Cite the paper 
%		'S.R. Dubey, S. Chakraborty, Average Biased ReLU Based CNN Descriptor for Improved Face Retrieval. arXiv Preprint, 1804.02051, 2018.' 
%	In case you are using this code.


function des=cnn_vgg_faces_abrelu(path_im,net)
%CNN_VGG_FACES  Demonstrates how to use VGG-Face

% Setup MatConvNet.
% setup;

% path_im = 'Alejandro_Toledo_0001.jpg';

% % Load the VGG-Face model. Avaliable at: http://www.vlfeat.org/matconvnet/pretrained/
% modelPath = fullfile('Pretrained','vgg-face.mat') ;
% % Load the model and upgrade it to MatConvNet current version.
% net = load(modelPath) ;
% net = vl_simplenn_tidy(net) ;

im=imread(path_im);
im1 = single(im) ; % note: 255 range
im2 = imresize(im1, net.meta.normalization.imageSize(1:2)) ;
im2 = bsxfun(@minus,im2,net.meta.normalization.averageImage) ;
res = vl_simplenn(net, im2);
des35=res(35).x;des35=(des35(:))';des35=des35-mean(des35);
% des36=res(36).x;des36=(des36(:))';
des35p=des35;des35p(des35p<0)=0;
des=des35p/sum(des35p);

end

