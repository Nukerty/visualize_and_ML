clear;
%%Data import
train_data = importdata("TwoLeadECG_TRAIN.txt");
test_data = importdata("TwoLeadECG_TEST.txt");

%%prepocessing Data
train_data(:,end+1) = train_data(:,1);
train_data(:,1) = [];
test_data(:,end+1) = test_data(:,1);
test_data(:,1) = [];


train_label=train_data(:,end);
test_label=test_data(:,end);
train_unlabled=train_data(:,1:end-1);
test_unlabled=test_data(:,1:end-1);

%%prior calculation
no_class = max(train_label);

for j = 1:no_class 
    ind = find(train_label == j); 
    p(j) = length(ind)/length(train_label); 
end 
%%
%feature extraction part



%

train_feature=train_unlabled;
test_feature=test_unlabled;

%%classfification
Label=[];miss=0;
t_case=size(test_feature,1);
for k=1:t_case
    Label(k) = Linear_classifier(train_feature,train_label,test_feature(k,:),no_class,p);
    if Label(k)~=test_label(k)
        miss=miss+1;
    end
end


Classification_rate = (1 - miss/t_case)*100;
fprintf("Classification rate : %f",Classification_rate);

