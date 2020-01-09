% Read the MNIST data into an array with images and labels
clearvars;
[images, labels] = readMNIST('t10kimages.idx3', 't10klabels.idx1',10000,0);

% Create the subsets
labelsSubset_train = find(labels == 1 | labels == 8, 900); %Vector mit den Labels von 1 und 6
imagesSubset = images(:,:,labelsSubset_train); %images sind 28x28

t = labels(labelsSubset_train);
t(t<=5) = 1;
t(t>5) = -1;
ClassA = find(t == +1); %Zeilenvektor mit den ersten 80
ClassB = find(t == -1);
X=[];
C=2; %Slack parameter

% Calculate the feature vectors for the imageSubset
for i=1:length(imagesSubset)
    binaryImage = im2bw(imagesSubset(:,:,i),0.2); %im2bw(I,level) converts the grayscale image I to binary image BW, by replacing all pixels in the input image with luminance greater than level with the value 1 (white) and replacing all other pixels with the value 0 (black).
    RegionProps= regionprops(binaryImage,'Solidity', 'FilledArea');
    X=[X; [RegionProps.Solidity, RegionProps.FilledArea]]; 
end

X(:,2) = X(:,2)/ max(X(:,2));

X=X';
t=t';

kernelfct=@rbfkernel;
[alpha, w0, w] = trainSVM(X,t,kernelfct);
S=find(alpha>0 & alpha<C);

% Plot 
Line = @(x1,x2) discriminant(alpha, X, t, x1, x2, w0, kernelfct);
LineA = @(x1,x2) discriminant(alpha, X, t, x1, x2, w0, kernelfct)+1;
LineB = @(x1,x2) discriminant(alpha, X, t, x1, x2, w0, kernelfct)-1;

figure;
%Datapoints with class-label
plot(X(1,ClassA),X(2,ClassA),'b.');
hold on;
plot(X(1,ClassB),X(2,ClassB),'r.');

%Support vectors
plot(X(1,S),X(2,S),'go','MarkerSize',9);

%window size
x1min = min(X(1,:))-0.5;
x1max = max(X(1,:))+0.5;
x2min = min(X(2,:))-0.5;
x2max = max(X(2,:))+0.5;

%Decision boundary
handle = ezplot(Line,[x1min x1max x2min x2max]);
set(handle,'Color','k','LineWidth',3); 

handleA = ezplot(LineA,[x1min x1max x2min x2max]);
set(handleA,'Color','k','LineWidth',1,'LineStyle',':');

handleB = ezplot(LineB,[x1min x1max x2min x2max]);
set(handleB,'Color','k','LineWidth',1,'LineStyle',':');

legend('Class A','Class B');