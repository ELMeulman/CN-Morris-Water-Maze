
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
ntrial=25;

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
si=0.16; % reach of place cell activation = 0.16 meter
f_grid=zeros(n_grid,n_grid);
f = @(xx,yy,ii) exp(-((xx-xi(ii)).^2+(yy-yi(ii)).^2)/(2*si^2))/sqrt(2*pi*si^2)/ni;
for ix=1:n_grid
    for iy=1:n_grid
        ff=f(X(ix,iy),Y(ix,iy),1:ni);
        f_grid(ix,iy)=ff(1);
    end
end
%surf(f_grid); % plot to see if it matches figure 2.

% Place place cells covering pool on grid, not exactly 493
%r2_grid=X.^2+Y.^2;
%ni=size(find(r2_grid<1),1);
%j=0;
%for ix=1:n_grid
%    for iy=1:n_grid
%        x=x_grid(ix);
%        y=y_grid(iy);
%        if x^2+y^2 <= 1
%            j=j+1;
%            xi(j)=x;
%            yi(j)=y;
%        end
%    end
%end

% Place place cells covering pool random;y, exactly 493
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
% Repeat function definition for placed place cells
f = @(xx,yy,ii) exp(-((xx-xi(ii)).^2+(yy-yi(ii)).^2)/(2*si^2));%/sqrt(2*pi*si^2)/ni;

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
theta=[0:1:7]/8*2*pi-pi; % directions for each of the 8 actions
zij=zeros(ni,nj);
zij_trial=zeros(ni,nj,ntrial);
ui_trial=zeros(ni,ntrial);
vi_trial=zeros(ni,ntrial);
aj=zeros(1,nj);
epsilon=0.1;
beta=2;

% CRITIC INFO
gamma=0.9975; % learning rate not given, need to play with this
w=zeros(ni,1); % weights for all place cells
w_trial=zeros(ni,ntrial);
C_grid=zeros(ntrial,n_grid,n_grid); % estimated value map (rows are y, columns are x) 

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
        aj=fi'*zij; % CC formula above (9), value for all dir
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
        C=w'*fi; % weights have changed so recalculate C for current location. 
        C_next=w'*fi_next; % weights have changed so recalculate C for next location. 
        reward=0; % zero reward away from platform
        if (x_rat(it+1,itrial)-x_platform)^2+(y_rat(it+1,itrial)-y_platform)^2 < dr_platform^2
            reward=1; % reward=1 on platform
            goal=1;
        end
        delta=reward+gamma*C_next-C; % formula (7)
        w=w+epsilon*delta*fi; % formula (8), update w for all neurons
                [mm,jj]=sort(abs(theta-tj)); % find direction after effect of momentum and bounce
        j_momentum=j;%jj(1);
        zij(:,j_momentum)=zij(:,j_momentum)+epsilon*delta*fi; % formula (10), update z for chosen action j = nearest angle ii(j,1)
        record(it,:)=[reward, C, C_next, delta, sum(epsilon*delta*fi)];
        if goal==1
            break;
        end
    end
    % update C_grid every trial, not every timestep which is expensive
    for ix=1:n_grid
        for iy=1:n_grid
            fi=f(X(ix,iy),Y(ix,iy),1:ni);
            C_grid(itrial,ix,iy)=w'*fi;
        end
    end
    C_grid(itrial,:,:)=C_grid(itrial,:,:)/max(max(C_grid(itrial,:,:)));
    w_trial(:,itrial)=w;%/max(w);
    zij_trial(:,:,itrial)=zij/max(max(zij));
    [mx,jmx]=max(zij,[],2);
    ui=sin(theta(jmx))';
    vi=cos(theta(jmx))';
    ui_trial(:,itrial)=ui;
    vi_trial(:,itrial)=vi;
    %sum(w)
    %itrial
end



% plot the place cells in the circular pool
figure;
scatter(xi,yi,abs(w)*1000);
%plot(xi, yi,'.');
hold on;
plot(pool_x,pool_y,'linewidth',2);
plot(x_platform, y_platform,'bo','MarkerFaceColor','b','Markersize',10);
plot(platform_x,platform_y,'b-','linewidth',2);
plot(x_rat(:,ntrial),y_rat(:,ntrial),'r-','linewidth',0.1);
plot(x_rat(1,ntrial),y_rat(1,ntrial),'bo','Markersize',30);
plot(x_rat(nt,ntrial),y_rat(nt,ntrial),'b*','Markersize',30);
Cp_plot=reshape(C_grid(ntrial,:,:),n_grid,n_grid);
contour(x_grid,y_grid,Cp_plot);
quiver(xi,yi,ui,vi);
axis equal;

figure;
Cp_2=reshape(C_grid(2,:,:),n_grid,n_grid);
Cp_7=reshape(C_grid(7,:,:),n_grid,n_grid);
Cp_22=reshape(C_grid(22,:,:),n_grid,n_grid);
axCp2=subplot(2,6,[1,2]); % value map after 2 trials
axCp7=subplot(2,6,[3,4]); % value map after 7 trials
axCp22=subplot(2,6,[5,6]); % value map after 22 trials
surf(axCp2,Cp_2);
surf(axCp7,Cp_7);
surf(axCp22,Cp_22);
zmax=max([max(max(max(C_grid(2,:,:)))),max(max(max(C_grid(7,:,:)))),max(max(max(C_grid(22,:,:))))]);
%zlim(axCp2,[-Inf zmax]);
%zlim(axCp7,[-Inf zmax]);
%zlim(axCp22,[-Inf zmax]);

axpa2=subplot(2,6,7); % preferred action map after 2 trials
axpa7=subplot(2,6,9); % preferred action map after 7 trials
axpa22=subplot(2,6,11); % preferred action map after 22 trials
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
scatter(axpa2,xi,yi,abs(w_trial(:,2))*1000);
scatter(axpa7,xi,yi,abs(w_trial(:,7))*1000);
scatter(axpa22,xi,yi,abs(w_trial(:,22))*1000);
%quiver(axpa2,x_grid,y_grid,DX_2,DY_2);
%quiver(axpa7,x_grid,y_grid,DX_7,DY_7);
%quiver(axpa22,x_grid,y_grid,DX_22,DY_22);
quiver(axpa2,xi,yi,ui_trial(:,2),vi_trial(:,2));
quiver(axpa7,xi,yi,ui_trial(:,7),vi_trial(:,7));
quiver(axpa22,xi,yi,ui_trial(:,22),vi_trial(:,22));
axpa2.DataAspectRatio=[1,1,1];
axpa7.DataAspectRatio=[1,1,1];
axpa22.DataAspectRatio=[1,1,1];
axis(axpa2,[-1 1 -1 1]);
axis(axpa7,[-1 1 -1 1]);
axis(axpa22,[-1 1 -1 1]);

axpath2=subplot(2,6,8); % path the rat takes in 2nd trial
axpath7=subplot(2,6,10); % path the rat takes in 7th trial
axpath22=subplot(2,6,12); % path the rat takes in 22th trial
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
plot(axpath2,x_rat(:,2),y_rat(:,2));
plot(axpath7,x_rat(:,7),y_rat(:,7));
plot(axpath22,x_rat(:,22),y_rat(:,22));
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
