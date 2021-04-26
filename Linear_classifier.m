function result= Linear_classifier(train,label,test,no_of_class,prior)
    max_value= -inf;
    for j= 1:no_of_class
        index= find(label== j);
        Data= train(index,:);
        m= mean(Data);
        v= cov(Data);
        g= -1/2*((test-m))*pinv(v)*(test-m)'+log(prior(j));
        if g>max_value
            max_value= g;
            count= j;
        end
    end
    result= count;
end