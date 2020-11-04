close all;clc;clear;
[D,C]=iris_dataset;
D=D';
D=D(:,1:2);
C=vec2ind(C)';
%For training data
n1=40;
n2=40;
%D1=D(1:50,:); %Class 1
D1=D(51:100,:);%Class2
D2=D(101:150,:); %Class 3
%First 30 sample for Training Data 
TrainD1=D1(1:n1,1:2); %Class 1
TrainD2=D2(1:n2,1:2); %Class 3
TrainData(1:n1,1:2)=TrainD1;
TrainData(n1+1:n1+n2,1:2)=TrainD2;
TrainTarget(1:n1,1)=1;
TrainTarget(n1+1:n1+n2,1)=2;
%Rest 20 sample for Testing Data
TestData(1:10,:)=D1(41:50,:);
TestData(11:20,:)=D2(41:50,:);
%Calculating Mean for both class and covariance of training data set
MeanC1=mean(TrainD1,1);
MeanC2=mean(TrainD2,1);
Covariance=cov(TrainData);
%Distance from Class1
InvCovariance=inv(Covariance);
Dis1=(TestData-MeanC1);
Distance1=diag((Dis1*(InvCovariance))*Dis1');
%Distance from Class2
Dis2=(TestData-MeanC2);
Distance2=diag((Dis2*(InvCovariance))*Dis2');
%taking two distance for comparison 
Distance(:,1)=Distance1;
Distance(:,2)=Distance2;
Distance=Distance';
%Comapring the distance to get the class label
[M,Class] = min(Distance);
Class=Class';
%Calculatig the Accuracy for testing dataset 
Acc1=0;
Acc2=0;
for i=1:size(Class,1)
    if(i<=10)
        if(Class(i,1)==1)
        Acc1=Acc1+1;
        end
    else
        if(Class(i,1)==2)
            Acc2=Acc2+1;
        end
    end
end
Accuracy=((Acc1+Acc2)/size(Class,1))*100;
%Scatter Ploting
%Cnew(1:10,1)=1;
%Cnew(11:20,1)=2;
%TestC=Class';
%TrainC=Cnew;
%TrainData=TrainData';
%TestData=TestData';
%gscatter(TestData(1,:),TestData(2,:),Cnew,'rb',[],[],'on','PL','PW');
%xlabel('PL');
%ylabel('PW');
%hold on;
%gscatter(TestData(1,:),TestData(2,:),TestC,'rb',[],[],'on','PL','PW');
%title('Min Distance(Mahalanobis) Based Classifier Test Output Data');
%title('Min Distance(Mahalanobis) Based Classifier Test Data');

ActualTestTarget(1:10,1)=1;
ActualTestTarget(11:20,1)=2;
TestData=TestData';

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
gscatter(TestData(1,:),TestData(2,:),Class,'bb','ox');
hold off
lgd = legend;
lgd.FontSize = 10;
lgd.Title.String = 'Data after testing';