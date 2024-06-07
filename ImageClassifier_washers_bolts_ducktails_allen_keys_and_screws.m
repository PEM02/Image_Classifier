clc;
clear;
close all;

% Ruta de la carpeta de imágenes y de la carpeta de destino para guardar imágenes recortadas
ruta_carpeta = "Bases Sossa\"; % Path of the database folder
ruta_nueva_carpeta = "ImagenesSegmentadas\"; % Path of the folder where segmented images will be saved

% Create the folder if it doesn't exist
if ~exist(ruta_nueva_carpeta, 'dir')
    mkdir(ruta_nueva_carpeta);
end

% Check if the descriptors file already exists
if ~exist("ImagenesSegmentadas\descriptors.mat")
    % Threshold for binarization
    umbral = 128;
    
    % Convert images to binary in the folder
    imagenes_binarias = convertir_imagenes_a_binarias_en_carpeta(ruta_carpeta, umbral);
    
    % Extract, crop, and save objects from each binary image
    for idx = 1:length(imagenes_binarias)
        objects = extractObjects(imagenes_binarias{idx});
        for j = 1:length(objects)
            % Save each cropped object in the new folder
            imwrite(objects{j}, fullfile(ruta_nueva_carpeta, sprintf('Objeto_%d_%d.png', idx, j)));
        end
    end
    
    % Call the processSegmentedImages function with the new folder path
    processSegmentedImages(ruta_nueva_carpeta);
end

% Open a dialog window for the user to select an image
[filename, pathname] = uigetfile({'*.jpg;*.bmp;*.png', 'Imagenes (*.jpg, *.bmp, *.png)'; '*.*', 'Todos los archivos (*.*)'}, 'Seleccione la imagen a clasificar');
if isequal(filename,0) || isequal(pathname,0)
    disp('Selección de imagen cancelada');
else
    imagePath = fullfile(pathname, filename);
    descriptorsPath = fullfile(ruta_nueva_carpeta, 'descriptors.mat'); % Path of the file to save descriptors
    result = classifyImage(imagePath, descriptorsPath); % Call the classifyImage function with the selected image and descriptors
end

% Display classification results
for i = 1:length(result)
    disp(result{i});
end

% Auxiliary functions

% Convert images to binary in a folder
function imagenes_binarias = convertir_imagenes_a_binarias_en_carpeta(ruta_carpeta, umbral)
    imagenes = leer_imagenes_en_carpeta(ruta_carpeta);
    numImagenes = length(imagenes);
    imagenes_binarias = cell(size(imagenes));
    for i = 1:numImagenes
        imagenes_binarias{i} = convertir_a_binaria(imagenes{i}, umbral);
    end
end

% Convert a single image to binary
function imagen_binaria = convertir_a_binaria(imagen, umbral)
    if size(imagen, 3) == 3
        imagen = rgb2gray(imagen);
    end
    imagen_binaria = imagen > umbral;
end

% Read images from a folder
function imagenes = leer_imagenes_en_carpeta(ruta_carpeta)
    archivos = dir(fullfile(ruta_carpeta, '*.BMP'));
    imagenes = cell(1, numel(archivos));
    for i = 1:numel(archivos)
        nombre_archivo = fullfile(ruta_carpeta, archivos(i).name);
        imagenes{i} = imread(nombre_archivo);
    end
end

% Extract objects from a binary image
function segmentedObjects = extractObjects(binaryImage)
    [rows, cols] = size(binaryImage);
    visited = false(rows, cols);
    segmentedObjects = {};
    numObjects = 0;
    
    % Preallocate memory for stack
    maxStackSize = numel(binaryImage);
    stack = zeros(maxStackSize, 2);
    stackIdx = 1;
    
    for r = 1:rows
        for c = 1:cols
            if binaryImage(r, c) == 1 && ~visited(r, c)
                numObjects = numObjects + 1;
                stack(stackIdx, :) = [r, c];
                stackSize = 1;
                objectImage = false(rows, cols);
                while stackSize > 0
                    pos = stack(stackSize, :);
                    stackSize = stackSize - 1;
                    x = pos(1);
                    y = pos(2);
                    if x > 0 && x <= rows && y > 0 && y <= cols && ~visited(x, y) && binaryImage(x, y)
                        visited(x, y) = true;
                        objectImage(x, y) = true;
                        if x > 1, stackSize = stackSize + 1; stack(stackSize, :) = [x-1, y]; end
                        if x < rows, stackSize = stackSize + 1; stack(stackSize, :) = [x+1, y]; end
                        if y > 1, stackSize = stackSize + 1; stack(stackSize, :) = [x, y-1]; end
                        if y < cols, stackSize = stackSize + 1; stack(stackSize, :) = [x, y+1]; end
                    end
                end
                croppedObject = cropImage(objectImage);
                segmentedObjects{numObjects} = croppedObject;
            end
        end
    end
    
    % Calculate the number of rows and columns for subplots
    numSubplots = ceil(sqrt(numObjects));
    numRows = numSubplots;
    numCols = numSubplots;
    
    % Create a new figure to show the subplots
    figure;
    
    % Show each segmented object in a subplot
    for i = 1:numObjects
        subplot(numRows, numCols, i);
        imshow(segmentedObjects{i});
        title(['figura ', num2str(i)]);
    end
end

% Crop a binary image to the bounding box of the object
function croppedImage = cropImage(binaryImage)
    [rows, cols] = find(binaryImage);
    croppedImage = binaryImage(min(rows):max(rows), min(cols):max(cols));
end

% Process segmented images to extract descriptors and cluster them
function processSegmentedImages(ruta_nueva_carpeta)
    archivos = dir(fullfile(ruta_nueva_carpeta, '*.png'));
    results = table();
    minArea = 50;
    minPerimeter = 30;
    numArchivos = length(archivos);
    
    % Preallocate memory for descriptors
    descriptors = zeros(numArchivos, 4);
    idx = zeros(numArchivos, 1);

    for i = 1:numArchivos
        imagen = imread(fullfile(ruta_nueva_carpeta, archivos(i).name));
        imagen = imagen > 0;
        area = calculateArea(imagen);
        perimeter = calculatePerimeter(imagen);
        eulerNumber = calculateEulerNumber(imagen);
        circularity = calculateCircularity(area, perimeter);
        if area >= minArea && perimeter >= minPerimeter
            results = [results; table({archivos(i).name}, area, perimeter, eulerNumber, circularity, 'VariableNames', {'ImageName', 'Area', 'Perimeter', 'EulerNumber', 'Circularity'})];
            descriptors(i, :) = [area, perimeter, eulerNumber, circularity];
        end
    end

    if height(results) > 0
        k = 5;
        [idx, C] = simpleClustering(descriptors, k);
        results.Group = idx;
        save(fullfile(ruta_nueva_carpeta, 'descriptors.mat'), 'descriptors', 'idx', 'C');

        for i = 1:k
            group_folder = fullfile(ruta_nueva_carpeta, sprintf('Grupo_%d', i));
            if ~exist(group_folder, 'dir')
                mkdir(group_folder);
            end
        end

        for i = 1:height(results)
            group_folder = fullfile(ruta_nueva_carpeta, sprintf('Grupo_%d', results.Group(i)));
            movefile(fullfile(ruta_nueva_carpeta, results.ImageName{i}), group_folder);
        end

        group_counts = containers.Map('KeyType', 'char', 'ValueType', 'double');

        % Iterate over each unique group to count the elements in each folder
        unique_groups = unique(results.Group);
        for j = 1:length(unique_groups)
            group_folder = fullfile(ruta_nueva_carpeta, sprintf('Grupo_%d', unique_groups(j)));
            files = dir(fullfile(group_folder, '*')); % Get all files in the folder
            files = files(~[files.isdir]); % Filter only files (exclude subfolders)
            group_counts(sprintf('Grupo_%d', unique_groups(j))) = numel(files);
        end
        
        % Print the number of elements in each group folder
        group_keys = keys(group_counts);
        for k = 1:length(group_keys)
            fprintf('La carpeta %s tiene %d elementos.\n', group_keys{k}, group_counts(group_keys{k}));
        end

        figure;
        scatter3(descriptors(:,1), descriptors(:,2), descriptors(:,3), 100, idx, 'filled');
        hold on;
        scatter3(C(:,1), C(:,2), C(:,3), 200, 'kx');
        hold off;
        title('Resultados del Clustering K-Means');
        xlabel('Área');
        ylabel('Perímetro');
        zlabel('Número de Euler');
        grid on;
    end

    writetable(results, fullfile(ruta_nueva_carpeta, 'image_analysis_results.csv'));
end

% Calculate the area of a binary image
function area = calculateArea(binaryImage)
    area = sum(binaryImage(:));
end

% Calculate the perimeter of a binary image
function perimeter = calculatePerimeter(binaryImage)
    [rows, cols] = size(binaryImage);
    perimeter = 0;
    for r = 1:rows-1
        for c = 1:cols-1
            if binaryImage(r, c) ~= binaryImage(r+1, c)
                perimeter = perimeter + 1;
            end
            if binaryImage(r, c) ~= binaryImage(r, c+1)
                perimeter = perimeter + 1;
            end
        end
    end
    perimeter = perimeter + sum(binaryImage(end, 1:end-1) ~= binaryImage(end, 2:end)) + sum(binaryImage(1:end-1, end) ~= binaryImage(2:end, end));
end

% Calculate the Euler number of a binary image
function E = calculateEulerNumber(binaryImage)
    % Label the objects in the original image
    [L, numObjects] = bwlabel(binaryImage, 8);
    
    % Invert the binary image
    invertedImage = ~binaryImage;
    
    % Label the holes in the inverted image
    [L_holes, numHoles] = bwlabel(invertedImage, 8);
    
    % Calculate the Euler number
    E = numObjects - numHoles;
end

% Calculate the circularity of an object
function circularity = calculateCircularity(area, perimeter)
    circularity = (4 * pi * area) / (perimeter^2);
end

% Perform simple clustering using K-means
function [idx, C] = simpleClustering(data, k)
    rng(42);
    C = data(randperm(size(data, 1), k), :);
    idx = zeros(size(data, 1), 1);
    oldIdx = zeros(size(data, 1), 1);
    iter = 0;
    maxIter = 100;

    while iter < maxIter
        for i = 1:size(data, 1)
            [~, idx(i)] = min(sum((data(i, :) - C).^2, 2));
        end

        for j = 1:k
            if any(idx == j)
                C(j, :) = mean(data(idx == j, :), 1);
            end
        end

        if all(idx == oldIdx)
            break;
        else
            oldIdx = idx;
        end

        iter = iter + 1;
    end
end

% Classify an image using precomputed descriptors
function classificationResults = classifyImage(imagePath, descriptorsPath)
    img = imread(imagePath);
    nuevo_tamano = [55, 55];
    [~, ~, ext] = fileparts(imagePath);
    ext = ext;
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    binaryImage = imbinarize(img_gray);
    if ~strcmpi(ext, '.BMP')
        binaryImage = imresize(binaryImage, nuevo_tamano);
    end
    segmentedObjects = extractObjects(binaryImage);
    classificationResults = {};
    minArea = 50;
    minPerimeter = 30;
    distanceThreshold = 200;
    distanceThresholdmaxi = 1e-6;
    
    data = load(descriptorsPath);
    descriptors = data.descriptors;
    idx = data.idx;
    centroids = data.C;

    countTornillos = 0;
    countColasDePato = 0;
    countAlcayatas = 0;
    countLlaves = 0;
    countRondanas = 0;
    countDesconocido = 0;

    for i = 1:length(segmentedObjects)
        objectImage = segmentedObjects{i};
        area = calculateArea(objectImage);
        perimeter = calculatePerimeter(objectImage);
        eulerNumber = calculateEulerNumber(objectImage);
        circularity = calculateCircularity(area, perimeter);
        
        % Filter small objects
        if area >= minArea && perimeter >= minPerimeter
            newImageDescriptor = [area, perimeter, eulerNumber, circularity];
            disp(['Nuevo descriptor de imagen: Área=', num2str(area), ', Perímetro=', num2str(perimeter), ', Número de Euler=', num2str(eulerNumber), ', Circularidad=', num2str(circularity)]);
            
            if i < 2
                tipo_distancia = input('¿Por cual distancia deseas agrupar?\n 1. Euclidiana\n 2. Mahalanobis\n 3. KNN\n Seleccione una opción: ');
            else
                tipo_distancia = 3;
            end
            % Classification based on the selected distance
            if tipo_distancia == 1
                group = EuclideanToCentroids(newImageDescriptor, centroids, distanceThreshold);
            elseif tipo_distancia == 2
                group = MahalanobisToCentroids(newImageDescriptor, descriptors, idx, distanceThreshold);
            elseif tipo_distancia == 3
                k = 3; % Number of nearest neighbors
                group = knn(newImageDescriptor, descriptors, idx, k, distanceThreshold);
            end

            % Assign the classification label
            switch group
                case 1
                    label = 'tornillos';
                    countTornillos = countTornillos + 1;
                case 2
                    label = 'colas de pato';
                    countColasDePato = countColasDePato + 1;
                case 3
                    label = 'alcayatas';
                    countAlcayatas = countAlcayatas + 1;
                case 4
                    label = 'llaves';
                    countLlaves = countLlaves + 1;
                case 5
                    label = 'rondanas';
                    countRondanas = countRondanas + 1;
                otherwise
                    label = 'desconocido';
                    countDesconocido = countDesconocido + 1;
            end

            classificationResults{end+1} = sprintf('La figura %d pertenece al grupo %d (%s)', i, group, label);
        end
    end
    fprintf('Conteo de tornillos: %d\n', countTornillos);
    fprintf('Conteo de colas de pato: %d\n', countColasDePato);
    fprintf('Conteo de alcayatas: %d\n', countAlcayatas);
    fprintf('Conteo de llaves: %d\n', countLlaves);
    fprintf('Conteo de rondanas: %d\n', countRondanas);
    fprintf('Conteo de desconocidos: %d\n', countDesconocido);
end

% Classification function using Euclidean distance
function group = EuclideanToCentroids(descriptor, centroids, distanceThreshold)
    % Calculate Euclidean distances between the descriptor and all centroids
    distances = sqrt(sum((centroids - descriptor).^2, 2));
    
    % Get the minimum distance
    [minDistance, group] = min(distances);
    
    % Check if the minimum distance is below the threshold
    if minDistance > distanceThreshold
        disp('Elemento desconocido');
        group = -1; % Assign a value to indicate an unknown element
    end
end

% Classification function using Mahalanobis distance
function group = MahalanobisToCentroids(descriptor, data, idx, distanceThreshold)
    % Calculate Mahalanobis distance between the descriptor and each class
    uniqueClasses = unique(idx);
    numClasses = length(uniqueClasses);
    mahalanobisDistances = zeros(numClasses, 1);
    
    for i = 1:numClasses
        classData = data(idx == uniqueClasses(i), :);
        mahalanobisDistances(i) = mahal(descriptor, classData);
    end
    
    % Get the minimum distance
    [minDistance, group] = min(mahalanobisDistances);
    
    % Check if the minimum distance is below the threshold
    if minDistance > distanceThreshold
        disp('Elemento desconocido');
        group = -1; % Assign a value to indicate an unknown element
    end
end

% Classification function using K-nearest neighbors
function group = knn(descriptor, descriptors, idx, k, distanceThreshold)
    % Calculate Euclidean distances between the descriptor and all descriptors
    distances = sqrt(sum((descriptors - descriptor).^2, 2));
    
    % Sort distances in ascending order and get the sorted indices
    [~, sortedIndices] = sort(distances);
    
    % Get the distance to the nearest neighbor
    nearestNeighborDistance = distances(sortedIndices(1));
    
    % Check if the nearest neighbor distance is below the threshold
    if nearestNeighborDistance > distanceThreshold
        disp('Elemento desconocido');
        group = -1; % Assign a value to indicate an unknown element
    else
        % Select the k nearest neighbors
        nearestNeighbors = idx(sortedIndices(1:k));
        
        % Determine the majority class among the nearest neighbors
        group = mode(nearestNeighbors);
    end
end
