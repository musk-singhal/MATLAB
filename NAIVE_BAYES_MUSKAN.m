close all;clc;clear;

%importing iris dataset
[D,C]=iris_dataset;
D=D';
D=D(:,1:4);
C=vec2ind(C)';

%Dimensionality reduction using PCA
%[Covariance,PC,EigValue,EigVector]=pca(D,'NumComponents',2);
PC(1:150,1)=D(:,1);
PC(1:150,2)=D(:,2);

n1=40;n2=40;
%LINEARLY SEPARABLE: CLASS 1 VS 3
%NON-LINEARLY SEPARABLE: CLASS 2 VS 3
%D1=PC(1:50,1:2); %Class 1
D1=PC(51:100,1:2); %Class 2
D2=PC(101:150,1:2); %Class 3

%TRAINING DATA
TrainTarget(1:n1,1)=1;
TrainTarget(n1+1:n1+n2,1)=2;
TrainD1=D1(1:n1,1:2); 
TrainD2=D2(1:n2,1:2); 
TrainData(1:n1,1:2)=TrainD1;
TrainData(n1+1:n1+n2,1:2)=TrainD2;
n=size(TrainData,1);
n1=size(TrainD1,1);
n2=size(TrainD2,1);
%PRIOR PROBABILITIES
Prior1=n1/n;
Prior2=n2/n;
%CLASS MEANS
MeanC1=mean(TrainD1,1);
MeanC2=mean(TrainD2,1);
%
Z1=(TrainD1-(1*MeanC1));
Z2=(TrainD2-(1*MeanC2));
%CLASS VARIANCE
Variance1=var(Z1);
Variance2=var(Z2);
%TESTING DATA
TestData(1:10,1:2)=D1(41:50,:);
TestData(11:20,1:2)=D2(41:50,:);
%Variance=(1/n1)*(Z1'*Z1);
%CovC1=cov(Z1);
F1x=mvnpdf(TestData,MeanC1,Variance1);
F2x=mvnpdf(TestData,MeanC2,Variance2);
%F3x=mvnpdf(D,MeanC1,CovC3);
Pc1x=F1x*Prior1;
Pc2x=F2x*Prior2;
%Pc3x=F3x*PC3;
PCX(:,1)=Pc1x;
PCX(:,2)=Pc2x;
%PCX(:,3)=Pc3x;
PCX=PCX';
[M,I] = max(PCX);
M=M';
I=I';
Acc1=0;
Acc2=0;
for i=1:20
    if(i<=10)
        if(I(i,1)==1)
        Acc1=Acc1+1;
        end
    else
        if(I(i,1)==2)
            Acc2=Acc2+1;
        end
    end
end

ActualTestTarget(1:10,1)=1;
ActualTestTarget(11:20,1)=2;
TestData=TestData';
Accuracy=((Acc1+Acc2)/size(I,1))*100;
%gscatter(TestData(1,:),TestData(2,:),Cnew,[],[],[],'on','PL','PW');

gscatter(TrainData(:,1),TrainData(:,2),TrainTarget,'rg');
xlabel('Sepal Length');ylabel('Sepal Width')
title("Actual Data");
hold on
gscatter(TestData(1,:),TestData(2,:),ActualTestTarget,'bb','ox');
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
gscatter(TestData(1,:),TestData(2,:),I,'bb','ox');
hold off
lgd = legend;
lgd.FontSize = 10;
lgd.Title.String = 'Data after testing';

