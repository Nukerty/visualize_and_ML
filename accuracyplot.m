function []=accuracyplot(train_unlabeled,train_label,test_unlabeled,test_label,no_class,p)
    acc=[];
    for k=1:80
        options = struct;
        options.ReducedDim=k;
        options.Kernel=0;
        [V,~]=KPCA(train_unlabeled',options);
        train_feature=train_unlabeled*V;
        test_feature=test_unlabeled*V;
        Label=[];miss=0;
        t_case=size(test_feature,1);
        for k=1:t_case
            Label(k) = Linear_classifier(train_feature,train_label,test_feature(k,:),no_class,p);
            if Label(k)~=test_label(k)
                miss=miss+1;
            end
        end
        cl_rate=(1 - miss/t_case)*100;
        acc=[acc cl_rate];
    end
    figure()
    plot([1:80],acc);
    xlabel("Reduced dimension");
    ylabel("Classification rate");
end