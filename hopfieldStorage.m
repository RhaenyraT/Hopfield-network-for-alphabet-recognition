num_iter=200;
how_many_alphabets=[25];
[alphabet] = prprob();
sequence=zeros(7,5);
alphabet(alphabet==0)=-1;
distortion=3;
Performance_error=zeros(length(how_many_alphabets),1);
 


%TRAIN ON BIG APLHABTETS AND RETREIEVE 

for kk=1:length(how_many_alphabets)

Tt=alphabet(:,1:how_many_alphabets(kk))';
Tn = Tt';

% RESHAPE 7 * 5 TO 35 * 25
for o=1:size(Tn,2)
    digit=reshape(Tn(:,o),5,7)' ;
    digitbig=imresize(digit,5);
    hopbig=imbinarize(digitbig,0.001);
    hopbig=double(hopbig);
    hopbig(hopbig==0)=-1;
    BIGundistorted(:,o)=reshape(hopbig,35*25,1) ;
end

% DISTORT 7 * 5 
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
    
    
% RESHAPE DISTORTED 7 * 5 TO 35 * 25
for o=1:size(Tn,2)
    digit=reshape(Tn(:,o),5,7)' ;
    digitbig=imresize(digit,5);
    hopbig=imbinarize(digitbig,0);
    hopbig=double(hopbig);
    hopbig(hopbig==0)=-1;
    BIGdistorted(:,o)=reshape(hopbig,35*25,1) ;
end

  BIGtrain=BIGundistorted;
  BIGtest=BIGdistorted;
  num_dig = how_many_alphabets(kk);
  net = newhop(BIGtrain);
  [Y,~,~] = sim(net,num_dig,[],BIGtrain);
  Y = Y';

    figure;
    subplot(3,1,1);
   Full_att=zeros(35,1);
    for i = 1:num_dig
    digit = Y(i,:);
    digit = reshape(digit,35,25); 

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

    subplot(3,1,2);
    Full_noise=zeros(35,1);
    for i = 1:num_dig
    digit = BIGdistorted(:,i);
    digit = reshape(digit,35,25);
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
    [Yn,~,~] = sim(net,{num_dig num_steps},{},BIGdistorted);
    Yn = Yn{1,num_steps};
    Yn = Yn';

subplot(3,1,3);
Full_recon=zeros(35,1);

    for i = 1:num_dig
    digit = Yn(i,:);
    digit = reshape(digit,35,25);
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