
% Computational Neuroscience
% week 10 DA99 

clear all

%    maze:
%
%    CRITIC              N     ACTOR
%                     NW | NE
%            C      W  \   /  E
%             \     \ SW   SE
%          wi  \     \ | S  /  zji
%               \     \| / /
% PLACE CELLS 1 2 ............ 493
%

rng(0); % set seed for repeatable random numbers

% PROBLEM SIZES
nj=8; % 8 actions = directions to choose from at any state
n_grid=25; % value map n_gridxn_grid

% TIMESTEPPING
T=120; % time out for each trial after 120 seconds
dt=0.1; % 0.1 second timesteps
nt=T/dt;
ntrial=250;

% POOL INFO
theta=[0:0.01:1]*2*pi;
pool_x = sin(theta);
pool_y = cos(theta);
theta_platform=1.75*pi;
r_platform=0.5;
x_platform = r_platform*sin(theta_platform); % mid platform position
y_platform = r_platform*cos(theta_platform);
dr_platform = 0.1/2; % platform radius 0.05 m (diameter is 0.1 m)
platform_x = x_platform+dr_platform*sin(theta);
platform_y = y_platform+dr_platform*cos(theta);

% GRID INFO
x_grid=[1:1:n_grid]*2/n_grid-1;
y_grid=[1:1:n_grid]*2/n_grid-1;
dgrid=2/n_grid;
[X,Y] = meshgrid(-1+dgrid:dgrid:1, -1+dgrid:dgrid:1);

% PLACECELL INFO
ni=493; % 493 place cells
xi=zeros(ni,1);
yi=zeros(ni,1);
thj=[0:1:7]/8*2*pi-pi;
si=0.16; % reach of place cell activation = 0.16 meter
sthj=pi/2;
f_grid=zeros(n_grid,n_grid);

% Place place cells covering pool randomly, exactly 493
j=0;
for i=1:ni*10
    x=rand*2-1;
    y=rand*2-1;
    r=x^2+y^2;
    if (j<ni)&&(r<1)
        j=j+1;
        xi(j)=x; % random for now, may need a better covering of map
        yi(j)=y; 
    end
end
if j<493
    print 'did not place all cells'
end
% Function definition for place cells firing rates
f = @(xx,yy,ii) exp(-((xx-xi(ii)).^2+(yy-yi(ii)).^2)/(2*si^2));
% Function definition for head direction cells firing rates
g = @(tt,jj) exp(-(tt-thj(jj)).^2)/(2*sthj^2);

% PLACEMAP INFO
f_map=zeros(n_grid);
for ix=1:n_grid
    for iy=1:n_grid
        ff=f(X(ix,iy),Y(ix,iy),1:ni);
        f_map(ix,iy)=f_map(ix,iy)+sum(ff);
    end
end
figure;
ax1=subplot(1,2,1);
surf(ax1,f_map);
axis square;
title('Place field');
ax2=subplot(1,2,2);
plot(ax2, pool_x, pool_y, 'r-','linewidth',2); 
hold on;
plot(ax2, xi, yi,'.');
contour(ax2, x_grid, y_grid, f_map);
axis equal;
title('Place cells in the pool');

% RAT INFO
v_rat=0.3; % swimming speed rat 0.3 m/s
x_rat=nan(nt,ntrial);
y_rat=nan(nt,ntrial);
theta_rat=nan(nt,ntrial); % swimming direction rat
rat_start_from_edge=0;
momentum=0.9;

% ACTOR INFO
theta=thj; % directions for each of the 8 actions
zikj=zeros(ni,nj,nj);
zikj_trial=zeros(ni,nj,nj,ntrial);
ui=zeros(ni,nj);
vi=zeros(ni,nj);
ui_trial=zeros(ni,ntrial);
vi_trial=zeros(ni,ntrial);
aj=zeros(1,nj);
epsilon=0.1;
beta=2;

% CRITIC INFO
gamma=0.9975; % learning rate not given, need to play with this
wi=zeros(ni,1); % weights for all place cells
wk=zeros(nj,1);
wik=zeros(ni,nj,1);

wi_trial=zeros(nj,ntrial);
wk_trial=zeros(nj,ntrial);
wik_trial=zeros(ni,nj,ntrial);
C_grid=zeros(ntrial,n_grid,n_grid,nj); % estimated value map (rows are y, columns are x) 

% Store itrial reward C C_next delta sum(epsilon*delta*fi)
record=zeros(nt, 5);
% TRIALS
for itrial=1:ntrial
    goal=0;
    t_rat=pi/2;%2*pi*(randi(4)-1)/4;%rand*2*pi; % position of rat starting anywhere along the edge of the pool
    x_rat(1,itrial)=(1-rat_start_from_edge)*sin(t_rat);
    y_rat(1,itrial)=(1-rat_start_from_edge)*cos(t_rat);
    theta_rat(1,itrial)=wrapToPi(t_rat+pi); % swimming direction of the rat swimming back towards the centre
    % TIMESTEPPING FOR SINGLE TRIAL
    for it=1:nt
        % ACTOR
        fi=f(x_rat(it,itrial),y_rat(it,itrial),1:ni); % formula (1), activity of each place cell for all dir
        gk=g(theta_rat(it,itrial),1:nj)'; 
        for j=1:nj
            aj(j)=fi'*zikj(:,:,j)*gk; % CC formula above (9), value for all dir
        end
        pj=exp(beta*aj)/sum(exp(beta*aj)); % formula (9)
        j=randsample(nj,1,true,pj); % actor's choice of direction
        % Momentum
        tj=wrapToPi(theta_rat(it,itrial)+(1-momentum)*theta(j)); % momentum 1:3 new direction : old direction
        xj=x_rat(it,itrial)+v_rat*dt*sin(tj); 
        yj=y_rat(it,itrial)+v_rat*dt*cos(tj);
        rj=sqrt(xj.^2+yj.^2);
        if rj>1 % bounce off edge of pool
            tj=wrapToPi(tj+pi); % for now turn rat by 180 degrees, although this is not real bounce...
        end
        x_rat(it+1,itrial)=x_rat(it,itrial)+v_rat*dt*sin(tj);
        y_rat(it+1,itrial)=y_rat(it,itrial)+v_rat*dt*cos(tj);
        theta_rat(it+1,itrial)=tj;
        % CRITIC
        fi_next=f(x_rat(it+1,itrial),y_rat(it+1,itrial),1:ni);
        gk_next=g(theta_rat(it+1,itrial),1:nj)';
        C=fi'*wik*gk; % weights have changed so recalculate C for current location. 
        C_next=fi_next'*wik*gk_next; % weights have changed so recalculate C for next location. 
        reward=0; % zero reward away from platform
        if (x_rat(it+1,itrial)-x_platform)^2+(y_rat(it+1,itrial)-y_platform)^2 < dr_platform^2
            reward=1; % reward=1 on platform
            goal=1;
        end
        delta=reward+gamma*C_next-C; % formula (7)
        wik=wik+epsilon*delta*fi*gk'; % formula (8), update w for all neurons
                %[mm,jj]=sort(abs(theta-tj)); % find direction after effect of momentum and bounce
        j_momentum=j;%jj(1);
        zikj(:,:,j)=zikj(:,:,j)+epsilon*delta*fi*gk'; % formula (10), update z for chosen action j = nearest angle ii(j,1)
        if goal==1
            break;
        end
    end
    % update C_grid every trial, not every timestep which is expensive
    for ix=1:n_grid
        for iy=1:n_grid
            for k=1:nj
                fi=f(X(ix,iy),Y(ix,iy),1:ni);
                gk=g(theta(j),1:nj)';
                C_grid(itrial,ix,iy,j)=fi'*wik*gk;
            end
        end
    end
    C_grid(itrial,:,:,:)=C_grid(itrial,:,:,:)/max(max(max(C_grid(itrial,:,:,:))));
    wik_trial(:,:,itrial)=wik;%/max(wi);
    zikj_trial(:,:,:,itrial)=zikj;
    for j=1:nj
        [mx,jmx]=max(reshape(zikj(:,j,:),ni,nj),[],2);
        ui(:,j)=sin(theta(jmx))';
        vi(:,j)=cos(theta(jmx))';
        ui_trial(:,itrial)=ui(:,j);
        vi_trial(:,itrial)=vi(:,j);
    end
    %sum(w)
    %itrial
end



% plot the place cells in the circular pool
figure;
%scatter(xi,yi,abs(w)*1000);
%plot(xi, yi,'.');
hold on;
plot(pool_x,pool_y,'linewidth',2);
plot(x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',10);
plot(platform_x,platform_y,'b-','linewidth',2);
plot(x_rat(:,ntrial),y_rat(:,ntrial),'r-','linewidth',0.1);
plot(x_rat(1,ntrial),y_rat(1,ntrial),'bo','Markersize',30);
plot(x_rat(nt,ntrial),y_rat(nt,ntrial),'b*','Markersize',30);
Cp_plot=reshape(sum(C_grid(ntrial,:,:,:),4),n_grid,n_grid);
contour(x_grid,y_grid,Cp_plot);
[DX,DY] = gradient(Cp_plot,1,1);
quiver(x_grid,y_grid,DX,DY);
%quiver(xi,yi,ui,vi);
axis equal;

figure;
for j=1:nj
ax(j)=subplot(3,3,j);
hold on;
plot(ax(j),pool_x,pool_y,'linewidth',2);
plot(ax(j),x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',10);
plot(ax(j),platform_x,platform_y,'b-','linewidth',2);
plot(ax(j),x_rat(:,ntrial),y_rat(:,ntrial),'r-','linewidth',0.1);
plot(ax(j),x_rat(1,ntrial),y_rat(1,ntrial),'bo','Markersize',30);
plot(ax(j),x_rat(nt,ntrial),y_rat(nt,ntrial),'b*','Markersize',30);
Cp_plot=reshape(sum(C_grid(ntrial,:,:,:),4),n_grid,n_grid);
contour(ax(j),x_grid,y_grid,Cp_plot);
quiver(ax(j),xi,yi,ui(:,j),vi(:,j));
axis equal;
end

figure;
Cp_2=reshape(sum(C_grid(20,:,:,:),4),n_grid,n_grid);
Cp_7=reshape(sum(C_grid(70,:,:,:),4),n_grid,n_grid);
Cp_22=reshape(sum(C_grid(220,:,:,:),4),n_grid,n_grid);
axCp2=subplot(3,3,1); % value map after 2 trials
axCp7=subplot(3,3,2); % value map after 7 trials
axCp22=subplot(3,3,3); % value map after 22 trials
surf(axCp2,Cp_2);
surf(axCp7,Cp_7);
surf(axCp22,Cp_22);
%zmax=max([max(max(max(C_grid(2,:,:)))),max(max(max(C_grid(7,:,:)))),max(max(max(C_grid(22,:,:))))]);
axis(axCp2,[0 25 0 25 -Inf 1]);
axis(axCp7,[0 25 0 25 -Inf 1]);
axis(axCp22,[0 25 0 25 -Inf 1]);
%zlim(axCp2,[-Inf 1]);
%zlim(axCp7,[-Inf 1]);
%zlim(axCp22,[-Inf 1]);

axpa2=subplot(3,3,4); % preferred action map after 2 trials
axpa7=subplot(3,3,5); % preferred action map after 7 trials
axpa22=subplot(3,3,6); % preferred action map after 22 trials
plot(axpa2,pool_x,pool_y,'linewidth',2);
plot(axpa7,pool_x,pool_y,'linewidth',2);
plot(axpa22,pool_x,pool_y,'linewidth',2);
hold(axpa2,'on');
hold(axpa7,'on');
hold(axpa22,'on');
plot(axpa2,x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',3);
plot(axpa7,x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',3);
plot(axpa22,x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',3);
plot(axpa2,platform_x,platform_y,'b-','linewidth',2);
plot(axpa7,platform_x,platform_y,'b-','linewidth',2);
plot(axpa22,platform_x,platform_y,'b-','linewidth',2);
[DX_2,DY_2] = gradient(Cp_2,1,1);
[DX_7,DY_7] = gradient(Cp_7,1,1);
[DX_22,DY_22] = gradient(Cp_22,1,1);
contour(axpa2,x_grid,y_grid,Cp_2); % commented out because no contour yet which gives error
contour(axpa7,x_grid,y_grid,Cp_7);
contour(axpa22,x_grid,y_grid,Cp_22);
%scatter(axpa2,xi,yi,abs(w_trial(:,2))*1000);
%scatter(axpa7,xi,yi,abs(w_trial(:,7))*1000);
%scatter(axpa22,xi,yi,abs(w_trial(:,22))*1000);
quiver(axpa2,x_grid,y_grid,DX_2,DY_2);
quiver(axpa7,x_grid,y_grid,DX_7,DY_7);
quiver(axpa22,x_grid,y_grid,DX_22,DY_22);
%quiver(axpa2,xi,yi,ui_trial(:,20),vi_trial(:,20));
%quiver(axpa7,xi,yi,ui_trial(:,70),vi_trial(:,70));
%quiver(axpa22,xi,yi,ui_trial(:,220),vi_trial(:,220));
axpa2.DataAspectRatio=[1,1,1];
axpa7.DataAspectRatio=[1,1,1];
axpa22.DataAspectRatio=[1,1,1];
axis(axpa2,[-1 1 -1 1]);
axis(axpa7,[-1 1 -1 1]);
axis(axpa22,[-1 1 -1 1]);

axpath2=subplot(3,3,7); % path the rat takes in 2nd trial
axpath7=subplot(3,3,8); % path the rat takes in 7th trial
axpath22=subplot(3,3,9); % path the rat takes in 22th trial
plot(axpath2,pool_x,pool_y,'linewidth',2);
plot(axpath7,pool_x,pool_y,'linewidth',2);
plot(axpath22,pool_x,pool_y,'linewidth',2);
hold(axpath2,'on');
hold(axpath7,'on');
hold(axpath22,'on');
plot(axpath2,x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',3);
plot(axpath7,x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',3);
plot(axpath22,x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',3);
plot(axpath2,platform_x,platform_y,'b-','linewidth',2);
plot(axpath7,platform_x,platform_y,'b-','linewidth',2);
plot(axpath22,platform_x,platform_y,'b-','linewidth',2);
plot(axpath2,x_rat(:,20),y_rat(:,20));
plot(axpath7,x_rat(:,70),y_rat(:,70));
plot(axpath22,x_rat(:,220),y_rat(:,220));
axpath2.DataAspectRatio=[1,1,1];
axpath7.DataAspectRatio=[1,1,1];
axpath22.DataAspectRatio=[1,1,1];
axis(axpath2,[-1 1 -1 1]);
axis(axpath7,[-1 1 -1 1]);
axis(axpath22,[-1 1 -1 1]);

        %tj=wrapToPi(theta_rat(it,itrial)+(1-momentum)*theta); % momentum 1:3 new direction : old direction
        %[TZ,TJ]=meshgrid(theta,tj); % directions tj are not the same as the 9 directions in theta
        %dT=abs(wrapToPi(TZ-TJ)); % difference between tj and each element of theta
        %[mm,ii]=sort(dT,2); % find theta closest to tj
        %zij_tj=zij(:,ii(:,1)); % pick up the zij value for the theta closest to tj
        %aj=tanh(fi'*zij_tj); % formula above (9), value for all dir
