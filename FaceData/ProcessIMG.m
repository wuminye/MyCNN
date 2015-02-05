
%初始化API
API_KEY = '68041207e5e5e968c82d31cae9424995';
API_SECRET = 'veqaq_gkK1krn1qZY-aU4fsk3VdAUxEG';
api = facepp(API_KEY, API_SECRET);


%获取文件列表
filedir = 'C:\Users\minye\Desktop\matlab\UFLDL\cnn\MyCNN\FaceData\pic\';
Files = dir(fullfile(filedir,'*'));
Files = Files(3:end);
LengthFiles = length(Files);
fprintf('<< %d  files totally>>\n',LengthFiles);
%创建保存区域
%Database.data = cell(LengthFiles,1);
%Database.cnt = 0;
load Database
fprintf('Database.cnt = %d  \n',Database.cnt);
be = Database.i+1;
for i = be:6625
    fprintf('\r[ %d ] processing\r',i);
    rst = detect_file(api,strcat(filedir,Files(i).name), 'all');
    close all;
    img_width = rst{1}.img_width;
    img_height = rst{1}.img_height;
    face = rst{1}.face;
    
    fprintf('Totally %d faces detected!\n', length(face));
    
    lm = cell(length(face),1);
    im = imread(strcat(filedir,Files(i).name));

    imshow(im);
    hold on;
    
    if length(face) == 0
        continue;
    end
    
    for k = 1 : length(face)
         % Draw face rectangle on the image
        face_i = face{k};
        lm{k} = api.landmark(face_i.face_id, '83p');
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
    
    pause(0.3);
    Database.cnt = Database.cnt +1;
    Database.data{Database.cnt}.filename = Files(i).name;
    Database.data{Database.cnt}.data = rst;
    Database.data{Database.cnt}.landmark = lm;
    if mod(Database.cnt,30)==0
       Database.i = i;
       save Database Database 
       fprintf('Saved.\n');
    end
end
fprintf('%d/%d Done...\n',cnt,LengthFiles);
save Database Database