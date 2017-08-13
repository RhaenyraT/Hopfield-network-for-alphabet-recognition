num_iter=15;
how_many_alphabets=[1:5];
[alphabet] = prprob();
sequence=zeros(7,5);
alphabet(alphabet==0)=-1;
distortion=10;
Performance_error=zeros(length(how_many_alphabets),1);
digitbiggie=zeros(length(how_many_alphabets),875);
NewT=zeros(length(how_many_alphabets),875);
for kk=1:length(how_many_alphabets)
    for r=1:how_many_alphabets(kk)
    digit=reshape(alphabet(:,r),5,7)' ;
    sequence=[sequence digit];
    digitbig=imresize(digit,5);
    digitbiggie=reshape(digitbig,875,1)' ;
    a=digitbiggie<0;m=digitbiggie>0;J=m-a;
    NewT(kk,:)=J;
    end
    %figure,imshow(sequence(:,6:end));

    %Tt=alphabet(:,1:how_many_alphabets(kk))';
    T=NewT'; %Tt';

    num_dig = how_many_alphabets(kk);
    net = newhop(T);
    [Y,~,~] = sim(net,num_dig,[],T);
    Y = Y';

    figure;
    subplot(num_dig,3,1);

    for i = 1:num_dig
    digit = Y(i,:);
    digit = reshape(digit,25,35)'; 

    subplot(num_dig,3,((i-1)*3)+1);
    imshow(digit)
    if i == 1
        title('Attractors')
    end
    hold on
    end


    %The plots show that they are attractors.

    %------------------------------------------------------------------------



    %Add noise to the digit maps

    Tn = NewT';
    for aa=1:how_many_alphabets(kk)
        for i=1:distortion
          temp_int=randi(size(alphabet,1));
             if Tn(temp_int,aa)==1
              Tn(temp_int,aa)=-1;
            else
              Tn(temp_int,aa)=1;
             end
        end
    end
    %Show noisy digits:

    subplot(num_dig,3,2);
     Tnn=Tn';
    for i = 1:num_dig
    digit = Tnn(i,:);
    digit = reshape(digit,25,35)';
    subplot(num_dig,3,((i-1)*3)+2);
    imshow(digit)
    if i == 1
        title('Noisy digits')
    end
    hold on
    end

    %------------------------------------------------------------------------

    %See if the network can correct the corrupted digits 


    num_steps = num_iter;

    Tn = Tnn';
    Tn = {Tn(:,1:how_many_alphabets(kk))};
    [Yn,~,~] = sim(net,{num_dig num_steps},{},Tn);
    Yn = Yn{1,num_steps};
    Yn = Yn';

    subplot(num_dig,3,3);

    for i = 1:num_dig
    digit = Yn(i,:);
    digit = reshape(digit,25,35)';
    subplot(num_dig,3,((i-1)*3)+3);
    imshow(digit)
    if i == 1
        title('Reconstructed noisy digits')
    end
    hold on
    end

Performance_error(kk)=sum(sum(Yn~=Y));  
%-----------------------------------------------------------------------
end

plot(1:max(how_many_alphabets),Performance_error,'-ro')
xlabel('Number of Patterns')
ylabel('Number of Pixel errors')
title('Hopfield Network Storage capacity, 10 pixel flips')