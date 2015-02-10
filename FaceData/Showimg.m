function [  ] = Showimg(data,dir)
%SHOWIMG Summary of this function goes here
%   Detailed explanation goes here
close all;
scale = 20;

F = imread([dir data.filename]);
subplot(2,2,1);
imshow(F);
hold on;
rst = data.data;
face = rst{1}.face;
img_width = rst{1}.img_width;
img_height = rst{1}.img_height;
for k = 1 : length(face)
         % Draw face rectangle on the image
        face_i = face{k};
        center = face_i.position.center;
        w = face_i.position.width / 100 * img_width;
        h = face_i.position.height / 100 * img_height;
        rectangle('Position', ...
            [center.x * img_width / 100 -  w/2, center.y * img_height / 100 - h/2, w, h], ...
            'Curvature', 0.4, 'LineWidth',2, 'EdgeColor', 'blue');
        % Detect facial key points
        ps = face_i.position;
        pt = ps.eye_left;
        scatter(pt.x * img_width / 100, pt.y * img_height / 100, 'g.');
        
        pt = ps.eye_right;
        scatter(pt.x * img_width / 100, pt.y * img_height / 100, 'g.');
        
        pt = ps.mouth_left;
        scatter(pt.x * img_width / 100, pt.y * img_height / 100, 'g.');
        
        pt = ps.mouth_right;
        scatter(pt.x * img_width / 100, pt.y * img_height / 100, 'g.');
        
        pt = ps.nose;
        scatter(pt.x * img_width / 100, pt.y * img_height / 100, 'g.');
end

Z = CalcGuass(data.data{1}, scale);

subplot(2,2,2);
%imshow(double(imresize(F,1/scale)).*Z/255);
imshow(Z);
subplot(2,2,3);
mesh(Z);
end

