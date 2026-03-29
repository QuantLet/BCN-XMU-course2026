
clc;
global t1;global t2;global s1;global s2;global d1;global d2;global w1; global w2; global n;global J;global h;global FLAG;
t1=0.24;t2=0.74;s1=0.58;s2=0.83;  %位置
d1=2.95;d2=1.6;w1=0.825;w2=-1.5;   %幅度
h=0.027;J=10;n=1024; %窗宽取0.027，分辨率取7（取大一点好），分辨率越小，变幅估计差得越远
FLAG=100;%迭代次数

filename = 'C:\Users\HCC\Desktop\RWTCd.csv';
A = csvread(filename);
Y = A(:,1);
Y = Y(1:1024);
Y = Y';
X = linspace(0,1,1025);
X = X(2:1025);  
%%%%%%%%%%%%%%%%%%%%%%%%%%% 第一次均值变点估计 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('第 1 次估计结果\n');
%T(x),m(x)的核密度估计
CM = kde0(X,0,1);
estDensity1 = kde1(X,Y,0,1);
estDensity2 = kde1(X,Y.^2,0,1);
coefficient1 = CoffJ(estDensity1(2,:));
CV1 = CVE(X,Y,0,1); 
statistic1 = TransformJ(coefficient1(2,:), CM, CV1);
fprintf('均值变点为：（第一行位置，第二行大小）');
MeanChange=ExtremePoint(statistic1(2,:),0,1);  
MeanChange(2,1) = ReturnCoff(MeanChange(:,1), CM, CV1 );
MeanChange(2,1) = ReturnValue(MeanChange(2,1),X,0,1);
MeanChange(2,2) = ReturnCoff(MeanChange(:,2), CM, CV1 );
MeanChange(2,2) = ReturnValue(MeanChange(2,2),X,0,1);
MeanChange

MeanPos1(1) = MeanChange(1,1);
MeanSize1(1) = MeanChange(2,1); 
MeanPos2(1) = MeanChange(1,2);
MeanSize2(1) = MeanChange(2,2);

%%%%%%%%%%%%%%%%%%%%%%%%% 第一次方差变点估计 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以均值变点为截断点重新划分子序列
j = 1; k = 1; l = 1;
for i = 1:n
    if X(i) <= MeanChange(1,1)
        subsample1(1,j) =  X(i);
        subsample1(2,j) =  sqrt(findValue(X(i),estDensity2)-(findValue(X(i),estDensity1))^2);
        j = j+1;
    elseif X(i) >= MeanChange(1,2)
        subsample3(1,l) =  X(i);
        subsample3(2,l) =  sqrt(findValue(X(i),estDensity2)-(findValue(X(i),estDensity1))^2);
        l = l+1;
    else
        subsample2(1,k) =  X(i);
        subsample2(2,k) =  sqrt(findValue(X(i),estDensity2)-(findValue(X(i),estDensity1))^2);
        k = k+1;
    end
end

%%%% 在子序列上各自估计方差变点，假设两类变点不重合 %%%%%%%%%%%%%%%%%
%核密度回归
CMsub1 = kde0(subsample1(1,:),0,MeanChange(1,1));
estDensityZsub1 = kde1(subsample1(1,:),subsample1(2,:),0,MeanChange(1,1));
coefficientZsub1 = Coff(estDensityZsub1(2,:),0,MeanChange(1,1)); 
CVsub1 = CVE(subsample1(1,:),subsample1(2,:),0,MeanChange(1,1));
statisticSub1 = Transform(coefficientZsub1(2,:), CMsub1, CVsub1,0,MeanChange(1,1));  
VolatilityChangeSub1 = ExtremePoint(statisticSub1(2,:),0,MeanChange(1,1));    
VolatilityChangeSub1(2,1) = ReturnCoff(VolatilityChangeSub1(:,1), CMsub1, CVsub1 );
VolatilityChangeSub1(2,1) = ReturnValue(VolatilityChangeSub1(2,1),subsample1,0,MeanChange(1,1)); 
VolatilityChangeSub1(2,2) = ReturnCoff(VolatilityChangeSub1(:,2), CMsub1, CVsub1 );
VolatilityChangeSub1(2,2) = ReturnValue(VolatilityChangeSub1(2,2),subsample1,0,MeanChange(1,1)); 

%核密度回归
CMsub2 = kde0(subsample2(1,:),MeanChange(1,1),MeanChange(1,2)); 
estDensityZsub2 = kde1(subsample2(1,:),subsample2(2,:),MeanChange(1,1),MeanChange(1,2));
coefficientZsub2 = Coff(estDensityZsub2(2,:),MeanChange(1,1),MeanChange(1,2)); 
CVsub2 = CVE(subsample2(1,:),subsample2(2,:),MeanChange(1,1),MeanChange(1,2));
statisticSub2 = Transform(coefficientZsub2(2,:), CMsub2, CVsub2,MeanChange(1,1),MeanChange(1,2)); 
VolatilityChangeSub2 = ExtremePoint(statisticSub2(2,:),MeanChange(1,1),MeanChange(1,2));       
VolatilityChangeSub2(2,1) = ReturnCoff(VolatilityChangeSub2(:,1), CMsub2, CVsub2 );
VolatilityChangeSub2(2,1) = ReturnValue(VolatilityChangeSub2(2,1),subsample2,MeanChange(1,1),MeanChange(1,2)); 
VolatilityChangeSub2(2,2) = ReturnCoff(VolatilityChangeSub2(:,2), CMsub2, CVsub2 );
VolatilityChangeSub2(2,2) = ReturnValue(VolatilityChangeSub2(2,2),subsample2,MeanChange(1,1),MeanChange(1,2)); 

%核密度回归
CMsub3 = kde0(subsample3(1,:),MeanChange(1,2),1); 
estDensityZsub3 = kde1(subsample3(1,:),subsample3(2,:),MeanChange(1,2),1);
coefficientZsub3 = Coff(estDensityZsub3(2,:),MeanChange(1,2),1); 
CVsub3 = CVE(subsample3(1,:),subsample3(2,:), MeanChange(1,2),1);
statisticSub3 = Transform(coefficientZsub3(2,:), CMsub3, CVsub3, MeanChange(1,2),1); 
VolatilityChangeSub3 = ExtremePoint(statisticSub3(2,:),MeanChange(1,2),1);    
VolatilityChangeSub3(2,1) = ReturnCoff(VolatilityChangeSub3(:,1), CMsub3, CVsub3 );
VolatilityChangeSub3(2,1) = ReturnValue(VolatilityChangeSub3(2,1),subsample3,MeanChange(1,2),1); 
VolatilityChangeSub3(2,2) = ReturnCoff(VolatilityChangeSub3(:,2), CMsub3, CVsub3 );
VolatilityChangeSub3(2,2) = ReturnValue(VolatilityChangeSub3(2,2),subsample3,MeanChange(1,2),1);

VolatilityChangeRaw = [VolatilityChangeSub1, VolatilityChangeSub2, VolatilityChangeSub3]; %第一行位置，第二行大小
    
%将方差跳幅最大的两个点作为方差变点的本次估计
volatilityfisrt = 0; volatilitysecond = 0; volatilitypos1 = 0; volatilitypos2 = 0; Vop1 = 0; Vop2 = 0; 
for i=1:6
    if (abs(VolatilityChangeRaw(2,i))>= volatilityfisrt) && (VolatilityChangeRaw(1,i)>=0.15) && (VolatilityChangeRaw(1,i)<=0.85)
        volatilityfisrt = abs(VolatilityChangeRaw(2,i));  
        volatilitypos1 = VolatilityChangeRaw(1,i);
        if VolatilityChangeRaw(2,i)>0
            Vop1 = 1;
        else 
            Vop1 = -1;
        end
    end
end
volatilityfisrt = volatilityfisrt *Vop1;
for i=1:6   %两变点靠太近，视为同一个变点
    if (abs(VolatilityChangeRaw(2,i))>=volatilitysecond ) && (abs(VolatilityChangeRaw(1,i)-volatilitypos1)>0.25 ) && (VolatilityChangeRaw(1,i)>=0.15) && (VolatilityChangeRaw(1,i)<=0.85)
        volatilitysecond = abs(VolatilityChangeRaw(2,i)); 
        volatilitypos2 = VolatilityChangeRaw(1,i);
        if VolatilityChangeRaw(2,i)>0
            Vop2 = 1;
        else 
            Vop2 = -1;
        end
    end
end

volatilitysecond = volatilitysecond *Vop2;
 
if volatilitysecond == 0
    volatilitysecond = volatilityfisrt;
    volatilitypos2 = volatilitypos1;
end
 
%方差变点按时间顺序排好
if volatilitypos2 < volatilitypos1
    x = volatilitypos1; y = volatilityfisrt; 
    volatilitypos1 = volatilitypos2; volatilityfisrt = volatilitysecond;
    volatilitypos2 = x; volatilitysecond = y;
end

VolatilityChange = [volatilitypos1, volatilitypos2; volatilityfisrt ,volatilitysecond];
VolatilityPos1(1) = VolatilityChange(1,1);
VolatilitySize1(1) = VolatilityChange(2,1);
VolatilityPos2(1) = VolatilityChange(1,2);
VolatilitySize2(1) = VolatilityChange(2,2);

fprintf('方差变点为：（第一行位置，第二行大小）');
VolatilityChange = [VolatilityPos1(1), VolatilityPos2(1); VolatilitySize1(1) , VolatilitySize2(1)]

criterion(1) = 10*( (t1-MeanPos1(1))^2 + (t2-MeanPos2(1))^2 + (s1-VolatilityPos1(1))^2 + (s2-VolatilityPos2(1))^2 ) + (d1-MeanSize1(1))^2 + (d2-MeanSize2(1))^2 + (w1-VolatilitySize1(1))^2 + (w2-VolatilitySize2(1))^2;  
fprintf('目标函数值为：%.2f\n\n', criterion(1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 算法迭代100次 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for flag = 2:FLAG
    fprintf('第 %d 次估计结果：\n', flag);
    
    %%%%%%%%%%%% 算法步骤1：估计均值变点，每个子段上估计一个变点 %%%%%%%%%%%%
    %以方差变点为截断点重新划分子序列
    j = 1; k = 1; l = 1;
    for i = 1:n
        if X(i) <= VolatilityChange(1,1)
            subsample1(1,j) =  X(i);
            subsample1(2,j) =  Y(i);
            j = j+1;
        elseif X(i) >= VolatilityChange(1,2)
            subsample3(1,l) =  X(i);
            subsample3(2,l) =  Y(i);
            l = l+1;
        else
            subsample2(1,k) =  X(i);
            subsample2(2,k) =  Y(i);
            k = k+1;
        end
    end
    
    %条件分支
    if volatilitypos1 == volatilitypos2   
        %%分成两端的时候，只有subsample1和subsample3，在子序列上各自估计均值变点
        CMsub1 = kde0(subsample1(1,:),0,VolatilityChange(1,1));
        estDensity1sub1 = kde1(subsample1(1,:), subsample1(2,:),0,VolatilityChange(1,1));
        coefficientsub1 = Coff(estDensity1sub1(2,:),0,VolatilityChange(1,1));  
        CVsub1 = CVE(subsample1(1,:),subsample1(2,:),0,VolatilityChange(1,1));
        statisticsub1 = Transform(coefficientsub1(2,:), CMsub1, CVsub1,0,VolatilityChange(1,1)); 
        MeanChangeSub1 = ExtremePoint(statisticsub1(2,:),0,VolatilityChange(1,1));
        MeanChangeSub1(2,1) = ReturnCoff(MeanChangeSub1(:,1), CMsub1, CVsub1 );
        MeanChangeSub1(2,1) = ReturnValue(MeanChangeSub1(2,1),subsample1,0,VolatilityChange(1,1)); 
        MeanChangeSub1(2,2) = ReturnCoff(MeanChangeSub1(:,2), CMsub1, CVsub1 );
        MeanChangeSub1(2,2) = ReturnValue(MeanChangeSub1(2,2),subsample1,0,VolatilityChange(1,1)); 
    
        CMsub3 = kde0(subsample3(1,:),VolatilityChange(1,2),1);
        estDensity1sub3 = kde1(subsample3(1,:), subsample3(2,:),VolatilityChange(1,2),1);
        coefficientsub3 = Coff(estDensity1sub3(2,:),VolatilityChange(1,2),1);  
        CVsub3 = CVE(subsample3(1,:),subsample3(2,:),VolatilityChange(1,2),1);
        statisticsub3 = Transform(coefficientsub3(2,:), CMsub3, CVsub3 ,VolatilityChange(1,2),1); 
        MeanChangeSub3 = ExtremePoint(statisticsub3(2,:),VolatilityChange(1,2),1);
        MeanChangeSub3(2,1) = ReturnCoff(MeanChangeSub3(:,1), CMsub3, CVsub3 );
        MeanChangeSub3(2,1) = ReturnValue(MeanChangeSub3(2,1),subsample3,VolatilityChange(1,2),1); 
        MeanChangeSub3(2,2) = ReturnCoff(MeanChangeSub3(:,2), CMsub3, CVsub3 );
        MeanChangeSub3(2,2) = ReturnValue(MeanChangeSub3(2,2),subsample3,VolatilityChange(1,2),1); 
        
        MeanChangeRaw = [MeanChangeSub1, MeanChangeSub3]  %第一行位置，第二行大小
        meanfirst = 0; meanpos1 = 0; meansecond = 0; meanpos2 = 0; Mop1 = 0; Mop2 = 0;   
        for i=1:4
            if (abs(MeanChangeRaw(2,i))>=meanfirst ) && (MeanChangeRaw(1,i) >=MeanPos1(1)-0.05) && (MeanChangeRaw(1,i) <=MeanPos1(1)+0.05)
                meanfirst = abs(MeanChangeRaw(2,i));
                meanpos1 = MeanChangeRaw(1,i);
                if MeanChangeRaw(2,i)>0
                    Mop1 = 1;
                else
                    Mop1 = -1;
                end
            end
        end 
        meanfirst = meanfirst *Mop1;
        for i=1:4   %两变点靠太近，间隔小于0.15视为同一个变点
            if (abs(MeanChangeRaw(2,i))>=meansecond ) && (abs(MeanChangeRaw(1,i)-meanpos1)>0.10 ) && (MeanChangeRaw(1,i) >=MeanPos2(1)-0.05 ) && (MeanChangeRaw(1,i) <=MeanPos2(1)+0.05)
                meansecond = abs(MeanChangeRaw(2,i));
                meanpos2 = MeanChangeRaw(1,i);
                if MeanChangeRaw(2,i)>0
                    Mop2 = 1;
                else
                    Mop2 = -1;
                end
            end
        end
        meansecond = meansecond *Mop2;
        
    else
        %%分成三段的时候，在子序列上各自估计均值变点
        CMsub1 = kde0(subsample1(1,:),0,VolatilityChange(1,1));
        estDensity1sub1 = kde1(subsample1(1,:), subsample1(2,:),0,VolatilityChange(1,1));
        coefficientsub1 = Coff(estDensity1sub1(2,:),0,VolatilityChange(1,1));
        CVsub1 = CVE(subsample1(1,:),subsample1(2,:),0,VolatilityChange(1,1));
        statisticsub1 = Transform(coefficientsub1(2,:), CMsub1, CVsub1,0,VolatilityChange(1,1)); 
        MeanChangeSub1 = ExtremePoint(statisticsub1(2,:),0,VolatilityChange(1,1));
        MeanChangeSub1(2,1) = ReturnCoff(MeanChangeSub1(:,1), CMsub1, CVsub1 );
        MeanChangeSub1(2,1) = ReturnValue(MeanChangeSub1(2,1),subsample1,0,VolatilityChange(1,1)); 
        MeanChangeSub1(2,2) = ReturnCoff(MeanChangeSub1(:,2), CMsub1, CVsub1 );
        MeanChangeSub1(2,2) = ReturnValue(MeanChangeSub1(2,2),subsample1,0,VolatilityChange(1,1)); 

        CMsub2 = kde0(subsample2(1,:),VolatilityChange(1,1),VolatilityChange(1,2));
        estDensity1sub2 = kde1(subsample2(1,:), subsample2(2,:),VolatilityChange(1,1),VolatilityChange(1,2));
        coefficientsub2 = Coff(estDensity1sub2(2,:),VolatilityChange(1,1),VolatilityChange(1,2)); 
        CVsub2 = CVE(subsample2(1,:),subsample2(2,:),VolatilityChange(1,1),VolatilityChange(1,2));
        statisticsub2 = Transform(coefficientsub2(2,:), CMsub2, CVsub2,VolatilityChange(1,1), VolatilityChange(1,2));  
        MeanChangeSub2 = ExtremePoint(statisticsub2(2,:),VolatilityChange(1,1),VolatilityChange(1,2));
        MeanChangeSub2(2,1) = ReturnCoff(MeanChangeSub2(:,1), CMsub2, CVsub2 );
        MeanChangeSub2(2,1) = ReturnValue(MeanChangeSub2(2,1),subsample2,VolatilityChange(1,1),VolatilityChange(1,2)); 
        MeanChangeSub2(2,2) = ReturnCoff(MeanChangeSub2(:,2), CMsub2, CVsub2 );
        MeanChangeSub2(2,2) = ReturnValue(MeanChangeSub2(2,2),subsample2,VolatilityChange(1,1),VolatilityChange(1,2)); 

        CMsub3 = kde0(subsample3(1,:),VolatilityChange(1,2),1);
        estDensity1sub3 = kde1(subsample3(1,:), subsample3(2,:),VolatilityChange(1,2),1);
        coefficientsub3 = Coff(estDensity1sub3(2,:),VolatilityChange(1,2),1);  
        CVsub3 = CVE(subsample3(1,:),subsample3(2,:),VolatilityChange(1,2),1);
        statisticsub3 = Transform(coefficientsub3(2,:), CMsub3, CVsub3 ,VolatilityChange(1,2),1);  
        MeanChangeSub3 = ExtremePoint(statisticsub3(2,:),VolatilityChange(1,2),1);
        MeanChangeSub3(2,1) = ReturnCoff(MeanChangeSub3(:,1), CMsub3, CVsub3 );
        MeanChangeSub3(2,1) = ReturnValue(MeanChangeSub3(2,1),subsample3,VolatilityChange(1,2),1); 
        MeanChangeSub3(2,2) = ReturnCoff(MeanChangeSub3(:,2), CMsub3, CVsub3 );
        MeanChangeSub3(2,2) = ReturnValue(MeanChangeSub3(2,2),subsample3,VolatilityChange(1,2),1); 
    
        MeanChangeRaw = [MeanChangeSub1, MeanChangeSub2, MeanChangeSub3] %第一行位置，第二行大小
    
        %更新均值变点估计
        meanfirst = 0; meanpos1 = 0; meansecond = 0; meanpos2 = 0; Mop1 = 0; Mop2 = 0;   
        for i=1:6
            if (abs(MeanChangeRaw(2,i))>=meanfirst ) && (MeanChangeRaw(1,i) >=MeanPos1(1)-0.05) && (MeanChangeRaw(1,i) <=MeanPos1(1)+0.05)
                meanfirst = abs(MeanChangeRaw(2,i));
                meanpos1 = MeanChangeRaw(1,i);
                if MeanChangeRaw(2,i)>0
                    Mop1 = 1;
                else
                    Mop1 = -1;
                end
            end
        end 
        meanfirst = meanfirst *Mop1;
        for i=1:6   %两变点靠太近，间隔小于0.15视为同一个变点
            if (abs(MeanChangeRaw(2,i))>=meansecond ) && (abs(MeanChangeRaw(1,i)-meanpos1)>0.10 ) && (MeanChangeRaw(1,i) >=MeanPos2(1)-0.05 ) && (MeanChangeRaw(1,i) <=MeanPos2(1)+0.05)
                meansecond = abs(MeanChangeRaw(2,i));
                meanpos2 = MeanChangeRaw(1,i);
                if MeanChangeRaw(2,i)>0
                    Mop2 = 1;
                else
                    Mop2 = -1;
                end
            end
        end
        meansecond = meansecond *Mop2; 
    end
    
    %判断是否在全局只探测出一个变点
    if meanpos1 == 0
        meanpos1 = meanpos2;
        meanfirst = meansecond;
    end
        
    if meanpos2 == 0  
        meanpos2 = meanpos1;
        meansecond = meanfirst;
    end   
        
    %优先遗传变幅大的点，再考虑距离远的点
    if meanpos1 == meanpos2
       meansecond = findAbsLarger(MeanChange(2,1), MeanChange(2,2));
       meanpos2 = findPos(meansecond, MeanChange); 
    end
    
    if abs(meanpos1-meanpos2)<0.10
        meansecond = findAbsSmaller(MeanChange(2,1), MeanChange(2,2));
        meanpos2 = findPos(meansecond, MeanChange);
    end    
    
    %最后，按位置顺序排好
    if meanpos2 < meanpos1
        x = meanpos1; y = meanfirst;
        meanpos1 = meanpos2; meanfirst = meansecond;
        meanpos2 = x; meansecond = y;
    end 
    
    MeanChange = [meanpos1, meanpos2; meanfirst, meansecond] 
    MeanPos1(flag) = MeanChange(1,1);
    MeanSize1(flag) = MeanChange(2,1);
    MeanPos2(flag) = MeanChange(1,2);
    MeanSize2(flag) = MeanChange(2,2);    
    
    %%%%%%%%%%%%% 算法步骤2：估计方差变点，每个子段上估计一个变点 %%%%%%%%%%%%
    %以均值变点为截断点重新划分子序列
    j = 1; k = 1; l = 1;
    for i = 1:n
        if X(i) <= MeanChange(1,1)
            subsample1(1,j) =  X(i);
            subsample1(2,j) =  sqrt(findValue(X(i),estDensity2)-(findValue(X(i),estDensity1))^2);
            j = j+1;
        elseif X(i) >= MeanChange(1,2)
            subsample3(1,l) =  X(i);
            subsample3(2,l) =  sqrt(findValue(X(i),estDensity2)-(findValue(X(i),estDensity1))^2);
            l = l+1;
        else
            subsample2(1,k) =  X(i);
            subsample2(2,k) =  sqrt(findValue(X(i),estDensity2)-(findValue(X(i),estDensity1))^2);
            k = k+1;
        end
    end
    
    %条件分支
    if meanpos1 == meanpos2  
         %核密度回归
        CMsub1 = kde0(subsample1(1,:),0,MeanChange(1,1)); 
        estDensityZsub1 = kde1(subsample1(1,:),subsample1(2,:),0,MeanChange(1,1));
        coefficientZsub1 = Coff(estDensityZsub1(2,:),0,MeanChange(1,1));  
        CVsub1 = CVE(subsample1(1,:),subsample1(2,:),0,MeanChange(1,1));
        statisticSub1 = Transform(coefficientZsub1(2,:), CMsub1, CVsub1 ,0,MeanChange(1,1));  
        VolatilityChangeSub1 = ExtremePoint(statisticSub1(2,:),0,MeanChange(1,1));
        VolatilityChangeSub1(2,1) = ReturnCoff(VolatilityChangeSub1(:,1), CMsub1, CVsub1 );
        VolatilityChangeSub1(2,1) = ReturnValue(VolatilityChangeSub1(2,1),subsample1,0,MeanChange(1,1));
        VolatilityChangeSub1(2,2) = ReturnCoff(VolatilityChangeSub1(:,2), CMsub1, CVsub1 );
        VolatilityChangeSub1(2,2) = ReturnValue(VolatilityChangeSub1(2,2),subsample1,0,MeanChange(1,1));
        
        %核密度回归
        CMsub3 = kde0(subsample3(1,:),MeanChange(1,2),1); 
        estDensityZsub3 = kde1(subsample3(1,:),subsample3(2,:),MeanChange(1,2),1);
        coefficientZsub3 = Coff(estDensityZsub3(2,:),MeanChange(1,2),1);  
        CVsub3 = CVE(subsample3(1,:),subsample3(2,:), MeanChange(1,2),1);
        statisticSub3 = Transform(coefficientZsub3(2,:), CMsub3, CVsub3 ,MeanChange(1,2),1);  
        VolatilityChangeSub3 = ExtremePoint(statisticSub3(2,:),MeanChange(1,2),1);    
        VolatilityChangeSub3(2,1) = ReturnCoff(VolatilityChangeSub3(:,1), CMsub3, CVsub3 );
        VolatilityChangeSub3(2,1) = ReturnValue(VolatilityChangeSub3(2,1),subsample3,MeanChange(1,2),1); 
        VolatilityChangeSub3(2,2) = ReturnCoff(VolatilityChangeSub3(:,2), CMsub3, CVsub3 );
        VolatilityChangeSub3(2,2) = ReturnValue(VolatilityChangeSub3(2,2),subsample3,MeanChange(1,2),1); 

        VolatilityChangeRaw = [VolatilityChangeSub1, VolatilityChangeSub3] 
        volatilityfisrt = 0; volatilitypos1 = 0; volatilitysecond = 0; volatilitypos2 = 0; Vop1 = 0; Vop2 = 0; 
        for i=1:4
            if (abs(VolatilityChangeRaw(2,i))>= volatilityfisrt ) && (VolatilityChangeRaw(1,i)>=0.15) && (VolatilityChangeRaw(1,i)<=0.85)
                volatilityfisrt = abs(VolatilityChangeRaw(2,i));
                volatilitypos1 = VolatilityChangeRaw(1,i); 
                if VolatilityChangeRaw(2,i)>0
                    Vop1 = 1;
                else
                    Vop1 = -1;
                end
            end
        end
        volatilityfisrt = volatilityfisrt *Vop1;
        
        for i=1:4  %间隔小于0.1视为同一个变点
            if (abs(VolatilityChangeRaw(2,i))>= volatilitysecond ) && (abs(VolatilityChangeRaw(1,i)-volatilitypos1)>0.25 ) && (VolatilityChangeRaw(1,i)>=0.15) && (VolatilityChangeRaw(1,i)<=0.85)
                volatilitysecond = abs(VolatilityChangeRaw(2,i));
                volatilitypos2 = VolatilityChangeRaw(1,i);
                if VolatilityChangeRaw(2,i)>0
                    Vop2 = 1;
                else
                    Vop2 = -1;
                end
            end
        end
        volatilitysecond = volatilitysecond *Vop2;
        
    else
        %分成三段时，在子序列上各自估计方差变点
        %核密度回归
        CMsub1 = kde0(subsample1(1,:),0,MeanChange(1,1)); 
        estDensityZsub1 = kde1(subsample1(1,:),subsample1(2,:),0,MeanChange(1,1));
        coefficientZsub1 = Coff(estDensityZsub1(2,:),0,MeanChange(1,1));  
        CVsub1 = CVE(subsample1(1,:),subsample1(2,:),0,MeanChange(1,1));
        statisticSub1 = Transform(coefficientZsub1(2,:), CMsub1, CVsub1 ,0,MeanChange(1,1)); 
        VolatilityChangeSub1 = ExtremePoint(statisticSub1(2,:),0,MeanChange(1,1));
        VolatilityChangeSub1(2,1) = ReturnCoff(VolatilityChangeSub1(:,1), CMsub1, CVsub1 );
        VolatilityChangeSub1(2,1) = ReturnValue(VolatilityChangeSub1(2,1),subsample1,0,MeanChange(1,1));
        VolatilityChangeSub1(2,2) = ReturnCoff(VolatilityChangeSub1(:,2), CMsub1, CVsub1 );
        VolatilityChangeSub1(2,2) = ReturnValue(VolatilityChangeSub1(2,2),subsample1,0,MeanChange(1,1));
    
        %核密度回归
        CMsub2 = kde0(subsample2(1,:),MeanChange(1,1),MeanChange(1,2)); 
        estDensityZsub2 = kde1(subsample2(1,:),subsample2(2,:),MeanChange(1,1),MeanChange(1,2));
        coefficientZsub2 = Coff(estDensityZsub2(2,:),MeanChange(1,1),MeanChange(1,2));  
        CVsub2 = CVE(subsample2(1,:),subsample2(2,:), MeanChange(1,1),MeanChange(1,2));
        statisticSub2 = Transform(coefficientZsub2(2,:), CMsub2, CVsub2,MeanChange(1,1),MeanChange(1,2));  
        VolatilityChangeSub2 = ExtremePoint(statisticSub2(2,:),MeanChange(1,1),MeanChange(1,2));    
        VolatilityChangeSub2(2,1) = ReturnCoff(VolatilityChangeSub2(:,1), CMsub2, CVsub2 );
        VolatilityChangeSub2(2,1) = ReturnValue(VolatilityChangeSub2(2,1),subsample2,MeanChange(1,1),MeanChange(1,2)); 
        VolatilityChangeSub2(2,2) = ReturnCoff(VolatilityChangeSub2(:,2), CMsub2, CVsub2 );
        VolatilityChangeSub2(2,2) = ReturnValue(VolatilityChangeSub2(2,2),subsample2,MeanChange(1,1),MeanChange(1,2)); 
    
        %核密度回归
        CMsub3 = kde0(subsample3(1,:),MeanChange(1,2),1); 
        estDensityZsub3 = kde1(subsample3(1,:),subsample3(2,:),MeanChange(1,2),1);
        coefficientZsub3 = Coff(estDensityZsub3(2,:),MeanChange(1,2),1); 
        CVsub3 = CVE(subsample3(1,:),subsample3(2,:), MeanChange(1,2),1);
        statisticSub3 = Transform(coefficientZsub3(2,:), CMsub3, CVsub3 ,MeanChange(1,2),1);  
        VolatilityChangeSub3 = ExtremePoint(statisticSub3(2,:),MeanChange(1,2),1);    
        VolatilityChangeSub3(2,1) = ReturnCoff(VolatilityChangeSub3(:,1), CMsub3, CVsub3 );
        VolatilityChangeSub3(2,1) = ReturnValue(VolatilityChangeSub3(2,1),subsample3,MeanChange(1,2),1); 
        VolatilityChangeSub3(2,2) = ReturnCoff(VolatilityChangeSub3(:,2), CMsub3, CVsub3 );
        VolatilityChangeSub3(2,2) = ReturnValue(VolatilityChangeSub3(2,2),subsample3,MeanChange(1,2),1); 
    
        VolatilityChangeRaw = [VolatilityChangeSub1, VolatilityChangeSub2, VolatilityChangeSub3]
    
        %更新方差变点估计
        volatilityfisrt = 0; volatilitypos1 = 0; volatilitysecond = 0; volatilitypos2 = 0; Vop1 = 0; Vop2 = 0; 
        for i=1:6
            if (abs(VolatilityChangeRaw(2,i))>= volatilityfisrt ) && (VolatilityChangeRaw(1,i)>=0.15) && (VolatilityChangeRaw(1,i)<=0.85)
                volatilityfisrt = abs(VolatilityChangeRaw(2,i)); %大小
                volatilitypos1 = VolatilityChangeRaw(1,i); %位置
                if VolatilityChangeRaw(2,i)>0
                    Vop1 = 1;
                else
                    Vop1 = -1;
                end
            end
        end
        volatilityfisrt = volatilityfisrt *Vop1;
        
        for i=1:6  %间隔小于0.1视为同一个变点
            if (abs(VolatilityChangeRaw(2,i))>= volatilitysecond ) && (abs(VolatilityChangeRaw(1,i)-volatilitypos1)>0.25 ) && (VolatilityChangeRaw(1,i)>=0.15) && (VolatilityChangeRaw(1,i)<=0.85)
                volatilitysecond = abs(VolatilityChangeRaw(2,i));
                volatilitypos2 = VolatilityChangeRaw(1,i);
                if VolatilityChangeRaw(2,i)>0
                    Vop2 = 1;
                else
                    Vop2 = -1;
                end
            end
        end
        volatilitysecond = volatilitysecond *Vop2;       
    end
    
    %判断是否在全局只探测出一个变点
    if volatilityfisrt == 0
        volatilitypos1 = volatilitypos2;
        volatilityfisrt = volatilitysecond;
    end
    
    if volatilitysecond == 0
        volatilitypos2 = volatilitypos1;
        volatilitysecond = volatilityfisrt;
    end
    
    %优先遗传变幅大的点，再考虑距离远的点
    if volatilitypos1 == volatilitypos2
       volatilitysecond = findAbsLarger(VolatilityChange(2,1), VolatilityChange(2,2));
       volatilitypos2 = findPos(volatilitysecond, VolatilityChange); %VolatilityChange是方差变点上期估计，优先遗传变幅较大的
    end
    
    if abs(volatilitypos1-volatilitypos2)<0.10
        volatilitysecond = findAbsSmaller(VolatilityChange(2,1), VolatilityChange(2,2));
        volatilitypos2 = findPos(volatilitysecond, VolatilityChange); 
    end
    
    %最后，方差变点按位置顺序排好
    if volatilitypos2 < volatilitypos1
       x = volatilitypos1; y = volatilityfisrt;
       volatilitypos1 = volatilitypos2; volatilityfisrt = volatilitysecond;
       volatilitypos2 = x; volatilitysecond = y;
    end     
    
    VolatilityChange = [volatilitypos1, volatilitypos2; volatilityfisrt ,volatilitysecond];
    VolatilityPos1(flag) = VolatilityChange(1,1);
    VolatilitySize1(flag) = VolatilityChange(2,1);
    VolatilityPos2(flag) = VolatilityChange(1,2);
    VolatilitySize2(flag) = VolatilityChange(2,2);
    VolatilityChange = [VolatilityPos1(flag), VolatilityPos2(flag); VolatilitySize1(flag), VolatilitySize2(flag)]
    criterion(flag) = 10*( (t1-MeanPos1(flag))^2 + (t2-MeanPos2(flag))^2 + (s1-VolatilityPos1(flag))^2 + (s2-VolatilityPos2(flag))^2 ) + (d1-MeanSize1(flag))^2 + (d2-MeanSize2(flag))^2 + (w1-VolatilitySize1(flag))^2 + (w2-VolatilitySize2(flag))^2;  
    fprintf('目标函数值：%.2f\n', criterion(flag));
    
end

%各指标作图
figure;
%subplot(2,2,1)
i=1:1:FLAG;
plot(i, MeanPos1(i), 'r', 'LineWidth', 1.5);
hold on;
plot(i, MeanPos2(i), 'm', 'LineWidth', 1.5);
hold off;
xlabel('迭代次数');ylabel('位置估计');
title('均值变点位置估计（优先遗传）');
legend('MeanPos1', 'MeanPos2');

figure;
%subplot(2,2,2)
i=1:1:FLAG;
plot(i, VolatilityPos1(i), 'c', 'LineWidth', 1.5);
hold on;
plot(i, VolatilityPos2(i), 'g', 'LineWidth', 1.5);
hold off;
xlabel('迭代次数');ylabel('位置估计');
title('方差变点位置估计（优先遗传）');
legend('VolatilityPos1', 'VolatilityPos2');

figure;
%subplot(2,2,3)
i=1:1:FLAG;
plot(i, MeanSize1(i), 'r', 'LineWidth', 1.5);
hold on;
plot(i, MeanSize2(i), 'm', 'LineWidth', 1.5);
hold off;
xlabel('迭代次数');ylabel('幅度估计');
title('均值变点幅度估计');
legend('MeanSize1', 'MeanSize2');

figure;
%subplot(2,2,4)
i=1:1:FLAG;
plot(i, VolatilitySize1(i), 'c', 'LineWidth', 1.5);
hold on;
plot(i, VolatilitySize2(i), 'g', 'LineWidth', 1.5);
hold off;
xlabel('迭代次数');ylabel('幅度估计');
title('方差变点幅度估计');
legend('VolatilitySize1', 'VolatilitySize2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 函数定义包 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 定义函数包
function sigma = volatility(var)%条件方差函数
    global s1;global s2;global w1; global w2;
        if var < s1
            sigma = 0.5;
        elseif var >= s2
            sigma = 0.5+w1+w2;
        else
            sigma = 0.5+w1;
        end
end
function ListSample=Rearrange(SampleX,SampleY) %X元素的冒泡排序
    A=SampleX; B=SampleY;
    N=length(A);
    for i=1:N
        for j=1:N-i
            if A(j)>A(j+1)
                temp = A(j);
                A(j) = A(j+1);
                A(j+1) = temp;
                temp = B(j);
                B(j) = B(j+1);
                B(j+1) = temp;
            end
        end
    end
    ListSample=[A;B];
end
function monwave = wave(x)   %小波函数不一定是对称的
    if (x>=1)&&(x<=2)
        monwave = 5*((x-1)^4);
    elseif (x>=-2)&&(x<=-1)
        monwave = (20/3)*((x+1)^3)+2*((x+1)^2);
    else 
        monwave = 0;
    end
end
function coefficient = CoffJ(SampleZ)  %求整个序列小波系数
    J = 6;
    a = 0; b = 1;
    N = length(SampleZ);
    pos = linspace(a, b, 2^J+1); 
    pos = pos(2:2^J+1);
    coff = zeros(1,2^J);
    for k=1:2^J
        total = 0;
            for i=1:N
                w = a + (i-1)*(b-a)/N;
                x = 2^J*(w-a)/(b-a) - (k-1);             
                total= total + wave(x) *SampleZ(i);
            end
        coff(k) = ((2^J)*(b-a))^(1/2)/N *total;
    end
    coefficient = [pos; coff];
end  
function coefficient = Coff(SampleZ,a,b)  %求子段小波系数
    global J; global n;
    N = length(SampleZ);
    Jj = floor(N/n*J);
    pos = linspace(a, b, 2^Jj+1); 
    pos = pos(2:2^Jj+1);
    coff = zeros(1,2^Jj);
    for k=1:2^Jj
        total = 0;
            for i=1:N
                w = a + (i-1)*(b-a)/N;
                x = 2^Jj*(w-a)/(b-a) - (k-1);             
                total= total + wave(x) *SampleZ(i);
            end
        coff(k) = ((2^Jj)*(b-a))^(1/2)/N *total;
    end
    coefficient = [pos; coff];
end  
function statisticM = TransformJ(SampleZ, CM, CV) %输入为Coefficient(2,:),CV；得到结果为小波系数经调整后的统计量（未取绝对值）
    J = 6;
    global n;
    pos = linspace(0, 1, n+1); 
    pos = pos(2:n+1);
    N = length(pos);
    res = zeros(1,N);
    
    for j=0:N-1  
        p = floor(2^J*(j/N))+1;
        res1(j+1) =sqrt(CM(2,j+1)) *  SampleZ(p)/sqrt(CV(2,j+1));%based on square estimator
        res2(j+1) =sqrt(CM(2,j+1)) *  SampleZ(p)/ CV(3,j+1);%based on absolute deviation estimator
        res3(j+1) =SampleZ(p)/ volatility(j/2^J);
    end
    statisticM = [pos; res1; res2; res3];
    %statisticM = [pos; res1];
end 
function statisticM = Transform(SampleZ, CM, CV ,a,b) %输入为Coefficient(2,:),CV；得到结果为小波系数经调整后的统计量（未取绝对值）
    global J; global n;
    pos = linspace(a, b, n+1); 
    pos = pos(2:n+1);
    N = length(pos);
    Jj = floor(N/n*J);
    res = zeros(1,N);
    
    for j=0:N-1  
        p = floor(2^Jj*(j/N))+1;
        res1(j+1) =sqrt(CM(2,j+1)) *  SampleZ(p)/sqrt(CV(2,j+1));%based on square estimator
        res2(j+1) =sqrt(CM(2,j+1)) *  SampleZ(p)/ CV(3,j+1);%based on absolute deviation estimator
        res3(j+1) =SampleZ(p)/ volatility(j/2^Jj);
    end
    statisticM = [pos; res1; res2; res3];
    %statisticM = [pos; res1];
end 
function estDensity0 = kde0(sampleX,a,b)   %输入为序列、区间端点
    global h; global n;
    N = length(sampleX);
    pos = linspace(a, b, n+1);
    pos = pos(2:n+1);
    x = repmat(pos, N, 1); %N*100
    X = repmat(sampleX', 1, length(pos));
    %num = sum(exp( -(x-sampleX).^2./(2*h^2) ).*sampleY, 1 ) ./ (sqrt(2*pi)*h*N);
    %den = sum(exp( -(x-sampleX).^2./(2*h^2) ), 1) ./ (sqrt(2*pi)*h*N);
    res = sum(    (abs((x-X)./h)<=1),   1 )./(2*h*N);
    estDensity0 = [pos; res];   
end
function estDensity1 = kde1(sampleX,sampleY,a,b)
    global h; global n;
    N = length(sampleX);
    pos = linspace(a, b, n+1);
    pos = pos(2:n+1);
    x = repmat(pos, N, 1); %N*100
    sampleX = repmat(sampleX', 1, length(pos));
    sampleY = repmat(sampleY', 1, length(pos));
    num = sum(exp( -(x-sampleX).^2./(2*h^2) ).*sampleY, 1 ) ./ (sqrt(2*pi)*h*N);
    den = sum(exp( -(x-sampleX).^2./(2*h^2) ), 1) ./ (sqrt(2*pi)*h*N);
    %num = sum(    (abs((x-sampleX)./h)<=1).*sampleY,   1 );
    %den = sum(    (abs((x-sampleX)./h)<=1),   1);
    res = num./den;
    estDensity1 = [pos; res];
end
function estDensity2 = kde2(sampleX,sampleY)
    global h; global n;
    % Compute the number of samples created
    N = length(sampleX);

    % Create a linearly spaced vector(100 grids)
    pos = linspace(0, 1, n+1);
    pos = pos(2:n+1);
    % Create two big matrices to avoid for loops
    x = repmat(pos, N, 1);
    sampleX = repmat(sampleX', 1, length(pos));
    sampleY = repmat(sampleY', 1, length(pos));
    
    %kernel regression
    num = sum(exp( -(x-sampleX).^2./(2*h^2) ).*(sampleY).^2, 1 ) ./ (sqrt(2*pi)*h*N);
    den = sum(exp( -(x-sampleX).^2./(2*h^2) ), 1) ./ (sqrt(2*pi)*h*N);
    res = num./den;
    
    % Form the output variable
    estDensity2 = [pos; res];
end
function CV = CVE(sampleX,sampleY,a,c) %估计方差函数，用局部线性平滑函数
    b = 0.02; 
    N = length(sampleX);
    pos = linspace(a, c, 1025);
    pos = pos(2:1025);
    Tnum = zeros(N);
    Tden = zeros(N);
    v = zeros(N,1024);
    T1 = zeros(N,1024);
    T2 = zeros(N,1024);
    T  = zeros(N);
    
    for j = 1:N
        for i = 1:N
            Tnum(j) = Tnum(j) + sampleY(i)*exp( -(sampleX(i)-sampleX(j))^2/(2*b^2) )/sqrt(2*pi);
            Tden(j) = Tden(j) + exp( -(sampleX(i)-sampleX(j))^2/(2*b^2) )/sqrt(2*pi);
        end
        T(j) = Tnum(j)/Tden(j);
    end
    
    for k = 1:1024
        for i = 1:N
            V(i,k) = exp( -(pos(k)-sampleX(i))^2/(2*b^2) )/sqrt(2*pi);
        end
    end
    
    for k = 1:1024
        for i = 1:N
            %v(i,k) = 0;
            %T1(i,k) = 0;
            %T2(i,k) = 0;
            for j = 1:N
                T1(i,k) = T1(i,k)+V(j,k)*(sampleX(j)-pos(k))^2;
                T2(i,k) = T2(i,k)+V(j,k)*(sampleX(j)-pos(k));
            end
            v(i,k) = V(i,k)*T1(i,k) - V(i,k)*(sampleX(i)-pos(k))*T2(i,k);
        end
    end
    
    for k =  1:1024
        num1(k) = 0;
        num2(k) = 0;
        den(k)  = 0;
        for i = 1:N
            num1(k) = num1(k)+v(i,k)*abs(sampleY(i)-T(i));
            num2(k) = num2(k)+v(i,k)*(sampleY(i)-T(i)).^2;
            den(k)  = den(k) +v(i,k);
        end
        ACV(k) = num1(k)/den(k)/sqrt(2/pi);
        SCV(k) = num2(k)/den(k);
    end
 
    CV = [pos; SCV; ACV];
end
function res = estM(sampleX,sampleY)
    b = 0.027;
    N = length(sampleX);
    x = repmat(sampleX, N, 1);
    X = repmat(sampleX', 1, N);
    Y = repmat(sampleY', 1, N);
    num = sum(    exp( -(x-X).^2./(2*b^2) ).*Y.^2 , 1 ) / (sqrt(2*pi)*b*N);
    den = sum(    exp( -(x-X).^2./(2*b^2) ) , 1 ) / (sqrt(2*pi)*b*N);
    res = num./den;    
end
function res = estT2(sampleX,sampleY)
    b = 0.027;
    N = length(sampleX);
    x = repmat(sampleX, N, 1);
    X = repmat(sampleX', 1, N);
    Y = repmat(sampleY', 1, N);
    num = sum(    exp( -(x-X).^2./(2*b^2) ).*Y , 1 ) / (sqrt(2*pi)*b*N);
    den = sum(    exp( -(x-X).^2./(2*b^2) ) , 1 ) / (sqrt(2*pi)*b*N);
    res = num.^2./den;   
end
%返回值为统计量绝对值的最大值点和对应的统计量值（不含绝对值）
function res = ExtremePoint(Series,a,b)  
    N = length(Series);
    L = b-a;
    i = 1;
    
    %注意调整精确度，即间距。注意区间两端估计要掐除。
    for mid = round(N*0.15): 1: round(N*0.85)    
        if (abs(Series(mid-1))< abs(Series(mid)) ) && (abs(Series(mid)) > abs(Series(mid+1)) ) 
            pos(i) = mid;
            jump(i) = Series(mid);
            i = i + 1;
            mid = mid + 1;
        end    
    end
       
    %也可能在区间上单调，没有极值点
    if i==1
        if abs(Series(round(N*0.15)))< abs(Series(round(N*0.85)))
            pos(i) = round(N*0.85);
            jump(i) = Series(round(N*0.85));
            p1 = round(N*0.85);
        else
            pos(i) = round(N*0.15);
            jump(i) = Series(round(N*0.15));
            p1 = round(N*0.15);
        end
        
        if jump(i)>0
            op1 = 1;
        else
            op1 = -1;
        end
        
    end
            
    preliminary = [pos; jump];  
    
    %寻找第一个绝对值极值点位置p1和大小first
    M = length(preliminary(1,:));
    first = 0;
    for i = 1:M
        if first < abs(preliminary(2,i))
            first = abs(preliminary(2,i));
            p1 = preliminary(1,i);  
            if preliminary(2,i) > 0
                op1 = 1;
            else
                op1 = -1;
            end
        end
    end
    %寻找次大值点位置p2和大小second
    second = 0;
    for i = 1:M
        if (second < abs(preliminary(2,i)) ) && (abs(preliminary(1,i)-p1)>0.1*N )
            second = abs(preliminary(2,i)); 
            p2 = preliminary(1,i);   
            if preliminary(2,i) > 0
                op2 = 1;
            else
                op2 = -1;
            end
        end
    end
    
    %只有一个极值点或者区间上单调的情形
    if second == 0
        p2 = p1;
        second = first;
        op2 = op1;
    end
    
    if p1>p2
        x = p1;     y = first;
        p1 = p2;    first = second;
        p2 = x;     second = y;
    end
    
    res = [a+p1/N*L, a+p2/N*L; first*op1, second*op2];    
end
function res = ExtremePoint1(Series,a,b) %求非可微函数绝对值的极值点
    N = length(Series);
    L = b-a;
    i = 1;
    
    %注意调整精确度，即间距。注意区间两端估计要掐除。
    for mid = round(N*0.15)+1: 1: round(N*0.85)-1     
        if (abs(Series(mid-1)) < abs(Series(mid)) ) && (abs(Series(mid)) > abs(Series(mid+1)) )
            pos(i) = mid;
            jump(i) = Series(mid);
            i = i + 1;
            mid = mid + 1;
        end    
    end
    preliminary = [pos; jump];  
    
    %寻找第一个极值点位置p1和大小first
    M = length(preliminary(1,:));
    first = 0;
    for i = 1:M
        if first < abs(preliminary(2,i))
            first = abs(preliminary(2,i));
            p1 = preliminary(1,i);   
            if preliminary(2,i) > 0
                op1 = 1;
            else
                op1 = -1;
            end
        end
    end
    res = [a+p1/N*L;  first*op1];
end
function res = ReturnCoff(MeanChangeX, CM, CV) %MeanChangeX为某个变点的（位置，大小）,输出为小波系数
    
    %查找变点位置对应的核回归值
    N = length(CM);
    low = 1; ptrCM = 0;
    for i = 1:N
        if abs(CM(1,i)-MeanChangeX(1))< low
            ptrCM = i;
            low = abs(CM(1,i)-MeanChangeX(1));
        end
    end
    den = sqrt(CM(2,ptrCM));
    
    N = length(CV);
    low = 1; ptrCV = 0;
    for i = 1:N
        if abs(CV(1,i)-MeanChangeX(1))< low
            ptrCV = i;
            low = abs(CV(1,i)-MeanChangeX(1));
        end
    end
    num = sqrt(CV(2,ptrCV));
    
    res = MeanChangeX(2)*num/den;
end
function res = ReturnValue(input,series,a,b) %输入为小波系数，输出为变幅估计值
    global J; global n;
    N = length(series);
    Jj = ceil(N/n*J);
    syms x;
    monwave = 5*((x-1)^4);
    den = (b-a)^(1/2) * 2^(-Jj/2) * int(monwave,x,1,2);
    res = input/den;
end
function res = TEstimation(Pos,estDensity)
    N = length(estDensity);
    low = 1; ptr = 0;
    for i = 1:N
        if abs(estDensity(1,i)-Pos)< low
            ptr = i;
            low = abs(estDensity(1,i)-Pos);
        end
    end
    
    res = estDensity(2, ptr);
end
function res = findAbsLarger(a,b)
    if abs(a)<=abs(b)
        res = b;
    else
        res = a;
    end
end
function res = findAbsSmaller(a,b)
    if abs(a)>=abs(b)
        res = b;
    else
        res = a;
    end
end
function res = findPos(x, Change)
    N = length(Change);
    for i = 1: N
        if x == Change(2, i)
            res = Change(1, i);
        end
    end 
end
function res = findArc(x, Change)
    N = length(Change);
    for i = 1: N
        if x == Change(1, i)
            res = Change(2, i);
        end
    end 
end
function res = findFurther(x, Change)
    if abs(x-Change(1,1))< abs(x-Change(1,2))
        res = Change(1,2);
    else 
        res = Change(1,1);
    end
end
function res = findValue(x,series)
    N = length(series);
    low = 10;
    for i = 1: N
        if abs(x-series(1,i))< low
            low = abs(x-series(1,i));
            ptr = i;
        end
    end
    res = series(2,ptr);
end