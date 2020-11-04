close all;clc;clear;
[D,C]=iris_dataset;
D=D';
D=D(:,1:2);
C=vec2ind(C)';
n1=40;
n2=40;
%D1=D(1:50,:); %Class 1
D1=D(51:100,:); %Class 2
D2=D(101:150,:); %Class 3
TrainD1=D1(1:n1,1:2); 
TrainD2=D2(1:n2,1:2); 
TrainData(1:n1,:)=TrainD1;
TrainData(n1+1:n1+n2,:)=TrainD2;
TestData(1:10,:)=D1(n1+1:50,:);
TestData(11:20,:)=D2(n2+1:50,:);
%TRAINING DATA
TrainTarget(1:n1,1)=1;
TrainTarget(n1+1:n1+n2,1)=2;
%Data(1:50,:)=D1;
%Data(51:100,:)=D2;
%Cnew(1:100,1)=1;
%Cnew(101:150,1)=2;
z=distmat(TestData,TrainData,'euclidean');
z=z';
[M,I] = min(z);
I=I';
Acc1=0;
Acc2=0;
for c=1:size(I,1)
    if (I(c,1)<=40)
        I(c,2)=1;
    else
        I(c,2)=2;
    end
end
Output=TestData;
Output(:,3)=I(:,2);
for i=1:20
    if(i<=10)
        if(Output(i,3)==1)
        Acc1=Acc1+1;
        end
    else
        if(Output(i,3)==2)
            Acc2=Acc2+1;
        end
    end
end
Accuracy=((Acc1+Acc2)/size(Output,1))*100;
TestC=Output(:,3);
%Cnew(1:20,1)=1;
%Cnew(21:40,1)=2;
%TrainC=Cnew;
ActualTestTarget(1:10,1)=1;
ActualTestTarget(11:20,1)=2;
%TestData=TestData';
%TrainData=TrainData';
%TestData=TestData';
%gscatter(TestData(1,:),TestData(2,:),Cnew,'rb',[],[],'on','PL','PW');
%xlabel('PL');
%ylabel('PW');
%hold on;
%gscatter(TestData(1,:),TestData(2,:),TestC,'rb',[],[],'on','PL','PW');
%title('Min Euclidean Distance Based Classifier Test Output');
%title('Min Euclidean Distance Based Classifier Test Data');


gscatter(TrainData(:,1),TrainData(:,2),TrainTarget,'rg');
xlabel('Sepal Length');ylabel('Sepal Width')
title("Actual Data");
hold on
gscatter(TestData(:,1),TestData(:,2),ActualTestTarget,'bb','ox');
hold off
lgd = legend;
lgd.FontSize = 10;
lgd.Title.String = 'Data to be tested';
%xlabel('PL');
%ylabel('PW');
%title('Naive Bayes Classifier Testing Data');
%title('Naive Bayes Classifier Test Output');
%gscatter(Data(1,:),Data(2,:),I,[],[],[],'on','PL','PW');


figure
gscatter(TrainData(:,1),TrainData(:,2),TrainTarget,'rg');
xlabel('Sepal Length');ylabel('Sepal Width')
title("Trained and Tested Data");
hold on
gscatter(TestData(:,1),TestData(:,2),I(:,2),'bb','ox');
hold off
lgd = legend;
lgd.FontSize = 10;
lgd.Title.String = 'Data after testing';








function dmat = distmat(a,varargin)
    
    
    % Set defaults
    method = 'euclidean';
    b = a;
    
    % Error check primary input
    if ~isnumeric(a)
        error('Expecting a matrix of floating point values for A input.');
    end
    
    % Process optional inputs
    for var = varargin
        arg = var{1};
        if ischar(arg)
            method = arg;
        elseif ~isempty(arg)
            b = arg;
        end
    end
    
    % Check input dimensionality
    [na,aDims] = size(a);
    [nb,bDims] = size(b);
    if (aDims ~= bDims)
        error('Input matrices must have the same dimensionality.');
    end
    
    % Create index matrices
    [j,i] = meshgrid(1:nb,1:na);
    
    % Compute array of inter-point differences
    delta = a(i,:) - b(j,:);
    
    % Compute distance by specified method
    dmat = zeros(na,nb);
    switch lower(method)
        case {'euclidean','euclid'}
            % Euclidean distance
            dmat(:) = sqrt(sum(delta.^2,2));
        case {'cityblock','city','block','manhattan','taxicab','taxi'}
            % Cityblock distance
            dmat(:) = sum(abs(delta),2);
        case {'chebyshev','cheby','chessboard','chess'}
            % Chebyshev distance
            dmat(:) = max(abs(delta),[],2);
        case {'grid','diag'}
            dmat(:) = max(abs(delta),[],2) + (sqrt(2) - 1)*min(abs(delta),[],2);
        otherwise
            error('Unrecognized distance method %s',method);
    end
    
end