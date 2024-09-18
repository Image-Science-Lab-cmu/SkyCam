clear all
close all


mirrors = ["Hyper", "Sphere"];

baseDir = './Synthetic_Slices_Data';

mirrors_AUC = [];
for ii=1:length(mirrors)
    mirror = mirrors(ii);
    Num_Days = 28;
    train_wid = 200; % Input window slice ( 200 = 100 mins)
    test_wid = 60;  % The window of slices to predict (60 = 30 minutes)
    
    Num_Slices = 800;
    GT = zeros(Num_Slices, test_wid, Num_Days);
    Preds = zeros(Num_Slices, test_wid, Num_Days);
    
    for dayNum = 1:Num_Days
        disp(dayNum)
        imgFname = sprintf('/%sCam_2023_2_%u.png', lower(mirror), dayNum);
        occFname = sprintf('%s_occlusion_state_2023_2_%u.mat', mirror, dayNum);
        img = imread(fullfile(baseDir, imgFname)); 
        img = double(img)/255;
        
        occ_val = load(fullfile(baseDir, occFname)).occ_state;
        
      
        rand_inds = random('unid', size(img, 2)-(train_wid+test_wid), Num_Slices, 1); 

        ang = getAngle(imgFname, occFname, baseDir);
        for ii = 1:Num_Slices
            st = rand_inds(ii);
            img_train = img(:, st+(1:train_wid), :); % Train image
            img_test = img(:, st+train_wid+(1:test_wid), :);    % Test image
            truth_occ_state = occ_val(:, st+train_wid+(1:test_wid), :); % The ground truth occlusion state for test window
            
            
            theta = ang*pi/180; 
            tau_min = 0; % min tau, (where we start at)
            tau_max = ((size(img, 1)-1)/2)/(tan(theta)); % max tau (How far we can look back in time based on theta)
            
            t0 = train_wid+(1:test_wid); % Indicies to predict
            tau = linspace(tau_min, tau_max, 200); % Selecting tau's to sample
            
            [T0, TAU] = meshgrid(t0, tau);
            XX = T0-TAU; % Sampling over tau's in x
            YY = -TAU*tan(theta)+(size(img, 1)+1)/2; % Sampling over the top part of the keogram in the y-direction
            
            for ch=1:3
                wrap_img(:,:,ch) = interp2(img_train(:, :, ch), XX, YY, 'linear', 0); % We compute the slice along each tau and stack them vertically (60)
            end
            
            
            occlusion_locs = find(truth_occ_state == 1);
            
            
            wrap_mn = mean(wrap_img, 3);
            mask = wrap_mn > 0;
            
            % (b-r)/(b+r)
            wrap_mn = (wrap_img(:,:,3)-wrap_img(:,:,1))./(1e-8+wrap_img(:,:,1)+wrap_img(:, :, 3));
            
            % Mean
            mn = sum(wrap_mn.*mask, 1)./(1e-8+sum(mask, 1));
            % Standard Deviation
            mn2 = sum((wrap_mn.^2).*mask, 1)./(1e-8+sum(mask, 1));
            st_d = sqrt(mn2-mn.^2);
            
            % To plot the ground truth on the plot
            occlusion_locs = find(truth_occ_state == 1);
            occlusion_locs_val_mean = mn(occlusion_locs);
            occlusion_locs_val_std = st_d(occlusion_locs);
            
            pred = mn; 
            truth = truth_occ_state;
        
            GT(ii, :, dayNum) = truth;
            Preds(ii, :, dayNum) =  pred;
        
        end
    end
    
    
    AUCs = zeros(test_wid, Num_Days, 1);
    for dayNum = 1:Num_Days
        for ii=1:test_wid
            curr_truth = GT(:, ii, dayNum);
            curr_pred = Preds(:, ii, dayNum);
            [falsePositiveRate, truePositiveRate, threshold, AUC, OPTROCPT] = perfcurve(curr_truth, 1-curr_pred, 1);
            
            best_thresh = threshold((falsePositiveRate==OPTROCPT(1))&(truePositiveRate==OPTROCPT(2)));
            AUCs(ii, dayNum) = AUC;
        end
    end
    
    
    figure
    AUCs_new = zeros(test_wid, 1);
    
    for ii=1:test_wid
        curr_truth = GT(:, ii, :);
        curr_truth = curr_truth(:);
        curr_pred = Preds(:, ii, :);
        curr_pred = curr_pred(:);
        
        [falsePositiveRate, truePositiveRate, threshold, AUC, OPTROCPT] = perfcurve(curr_truth, 1-curr_pred, 1);
        subplot(10,6,ii)
        plot(falsePositiveRate, truePositiveRate)
        title(sprintf('t+%u', ii))
        
        best_thresh = threshold((falsePositiveRate==OPTROCPT(1))&(truePositiveRate==OPTROCPT(2)));
        AUCs_new(ii) = AUC;
    end
    
    
    figure
    AUCs = mean(AUCs, 2);
    title(sprintf('%s | Time Vs AUC | %u Samples', mirror, Num_Slices))
    ylabel('AUC Value')
    xlabel('Time +')
    hold on 
    plot(AUCs_new)
    hold off
        

    mirrors_AUC = [mirrors_AUC AUCs_new(:)];
end



% Plot the AUCS for each mirror on one plot
plot(mirrors_AUC(:,1))
title(sprintf('Time Vs AUC | %u Samples', Num_Slices))
ylabel('AUC Value')
xlabel('Time +')
hold on 
plot(mirrors_AUC(:,2))
legend(mirrors(1), mirrors(2))
hold off

mirror1_auc = mirrors_AUC(:,1);
mirror2_auc = mirrors_AUC(:,2);



function optimal_angle = getAngle(imgName, occName, baseDir)
    STDS = [];
    THETAS = [];
    for ang=65:0.5:85

        
        img = imread(fullfile(baseDir, imgName)); 
        img = double(img)/255;
        
        occ_val = load(fullfile(baseDir, occName)).occ_state;
        
        train_wid = 200; % Input window slice ( 200 = 100 mins)
        test_wid = 60;  % The window of slices to predict (60 = 30 minutes)
        
        st = random('unid', size(img, 2)-(train_wid+test_wid), 1, 1); % Selecting random windows in the keogram
        st = 100;
        img_train = img(:, st+(1:train_wid), :); % Train image
        img_test = img(:, st+train_wid+(1:test_wid), :);    % Test image
        truth_occ_state = occ_val(:, st+train_wid+(1:test_wid), :); % The ground truth occlusion state for test window
        
       
        theta = ang*pi/180; 
        tau_min = 0; % min tau, (where we start at)
        tau_max = ((size(img, 1)-1)/2)/(tan(theta)); % max tau (How far we can look back in time based on theta)
        
        t0 = train_wid+(1:test_wid); % Indicies to predict
        tau = linspace(tau_min, tau_max, 200); % Selecting tau's to sample
        
        [T0, TAU] = meshgrid(t0, tau);
        XX = T0-TAU; % Sampling over tau's in x
        YY = -TAU*tan(theta)+(size(img, 1)+1)/2; % Sampling over the top part of the keogram in the y-direction
        
        for ch=1:3
            wrap_img(:,:,ch) = interp2(img_train(:, :, ch), XX, YY, 'linear', 0); % We compute the slice along each tau and stack them vertically (60)
        end
        
        
        occlusion_locs = find(truth_occ_state == 1);

        wrap_mn = mean(wrap_img, 3);
        mask = wrap_mn > 0;
        
        
        % (b-r)/(b+r)
        wrap_mn = (wrap_img(:,:,3)-wrap_img(:,:,1))./(1e-8+wrap_img(:,:,1)+wrap_img(:, :, 3));
        
        % Mean
        mn = sum(wrap_mn.*mask, 1)./(sum(mask, 1));
        % Standard Deviation
        mn2 = sum((wrap_mn.^2).*mask, 1)./(sum(mask, 1));
        st_d = sqrt(mn2-mn.^2);

        STDS = [STDS mean(st_d, "omitnan")];
        THETAS = [THETAS ang];
    end

    min_idx = find(STDS == min(STDS));
    optimal_angle =  THETAS(min_idx);

end











