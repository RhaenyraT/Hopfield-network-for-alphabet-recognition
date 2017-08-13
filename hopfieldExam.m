num_iter=300;
how_many_alphabets=[5];
[alphabet] = prprob();
sequence=zeros(7,5);
alphabet(alphabet==0)=-1;
distortion=3;
Performance_error=zeros(length(how_many_alphabets),1);

for kk=1:length(how_many_alphabets)
%     for r=1:how_many_alphabets(kk)
%     digit=reshape(alphabet(:,r),5,7)' ;
%     sequence=[sequence digit];
%     end
%     figure,imshow(sequence(:,6:end));

    Tt=alphabet(:,1:how_many_alphabets(kk))';
    T=Tt';

    num_dig = how_many_alphabets(kk);
    net = newhop(T);
    [Y,~,~] = sim(net,num_dig,[],T);
    Y = Y';

    figure;
    subplot(3,1,1);
   Full_att=zeros(7,1);
    for i = 1:num_dig
    digit = Y(i,:);
    digit = reshape(digit,5,7)'; 

    Full_att=[Full_att digit];
    if i == 1
        title('Attractors')
    end
    hold on
    end
    subplot(3,1,1);
    imshow(Full_att)
     
    %The plots show that they are attractors.

    %------------------------------------------------------------------------



    %Add noise to the digit maps

    Tn = Tt';
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

    subplot(3,1,2);
     Tnn=Tn';
       Full_noise=zeros(7,1);
    for i = 1:num_dig
    digit = Tnn(i,:);
    digit = reshape(digit,5,7)';
      Full_noise=[Full_noise digit];

    
    if i == 1
        title('Noisy digits')
    end
    hold on
    end
    subplot(3,1,2);
    imshow(Full_noise)
    
    %------------------------------------------------------------------------

    %See if the network can correct the corrupted digits 


    num_steps = num_iter;

    Tn = Tnn';
    Tn = {Tn(:,1:how_many_alphabets(kk))};
    [Yn,~,~] = sim(net,{num_dig num_steps},{},Tn);
    Yn = Yn{1,num_steps};
    Yn = Yn';

subplot(3,1,3);
Full_recon=zeros(7,1);

    for i = 1:num_dig
    digit = Yn(i,:);
    digit = reshape(digit,5,7)';
    Full_recon=[Full_recon digit];
      if i == 1
        title('Reconstructed noisy digits')
    end
    hold on
    end
  subplot(3,1,3);
    imshow(Full_recon)
    
Performance_error(kk)=sum(sum(Yn~=Y));  
%-----------------------------------------------------------------------
end

figure;plot(1:max(how_many_alphabets),Performance_error,'-ro')
xlabel('Number of Patterns')
ylabel('Number of Pixel errors')
title('Hopfield Network Storage capacity, 3 pixel flips')