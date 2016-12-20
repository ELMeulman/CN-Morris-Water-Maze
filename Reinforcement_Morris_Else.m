%set space
radius = 1;             %radius of the total space
r_pl = [0.05, 0.05];    %radius of platform
p_pl = [-0.4, 0.4];     %place of platform
speed = 0.3;            %swimming speed per s
dt = 0.1;
max_t_steps = 120/dt;   %timeout after 120 s
momentum = 3;           %actor:previous direction is 1:3 
p_start = [[0,-radius];[radius,0];[0,radius];[-radius,0]]; %NESW options
dist = speed*dt;        %distance moved in one time step

%vector of directions moved in one time step, north, northeast, etc.
n_directions = 8;
directions = zeros(n_directions,2);
dir_angle = linspace(2*pi/n_directions, 2*pi, n_directions);
for i = 1:n_directions; 
    directions(i,:) = [cos(dir_angle(i)), sin(dir_angle(i))]; %unit length
end       

%place cells
N = 493; %number of place cells
sigma = 0.16; %breadth of the field
%position of center of the place cells
%TO DO: spread uniformly
theta = rand(1,N)*(2*pi);
r = radius*sqrt(rand(1,N));
x = r.*cos(theta);
y = r.*sin(theta);
s = [x;y]'; %i x,y position of place cells

%init variables
z = zeros(8,N); %ji, weight from place cell i to action cell j
w = zeros(1,N); % weight from place cell i

%functions
%activity of place cell i at position p
F_f_i = @(p,i) exp(-( (sqrt((p(1)-s(i,1)).^2+(p(2)-s(i,2)).^2)) .^2) / (2*sigma^2) );

%parameters
n_trials = 22;
gamma = 0.9975; %discounting factor from DA
eps = 0.5;

dx = 0.05;
x = -1:dx:1;
y = -1:dx:1;
C_logger = zeros(n_trials,size(x,2),size(y,2));
p_logger = nan(n_trials,max_t_steps,2);
h_logger = zeros(max_t_steps,2);
pa_logger = zeros(n_trials,size(x,2),size(y,2),2);

for tr = 1:n_trials
    trialend = false;
    R = 0; %reward
    p = p_start(randi([1 4]),:); %start randomly at one of the start pos
    prev_heading = [0,0]; %initialize previous heading
    t = 1;
    
    while ~trialend
        %determine action
        f_i = F_f_i(p,1:N); %firing rates
        a_j = z(:,:)*f_i; %activity of action cell
        
%         if sum(exp(2*a_j)) == Inf
%             a_j = a_j/100000; %can't calculate probs if a_j is too large
%         end
%         if sum(exp(2*a_j)) == 0
%             P_j = ones(size(P,1),1); %can't calculate probs if all negative
%         else
            
        P_j = exp(2*a_j)/sum(exp(2*a_j)); %probs of directions
%         end
        action = find(cumsum(P_j)-rand(1) >= 0, 1);
        
        %position after action
        heading = (directions(action,:) + momentum * prev_heading)/(momentum+1);
        p_new = p + (heading /(sqrt(heading(1)^2+heading(2)^2)/dist)); %move with size dist
        %reflective walls
        if sqrt(p_new(1)^2+p_new(2)^2) >= radius
            heading = - heading; %move in opposite direction
            p_new = p + (heading /(sqrt(heading(1)^2+heading(2)^2)/dist));
        end
        
        %check if goal is reached
        if sqrt((p_new(1)-p_pl(1))^2+(p_new(2)-p_pl(2))^2) <= r_pl(1)        
            R = 1;
            trialend = true;
        end
        
        %update actor and critic
        C_p = w*f_i;
        C_p_new = w*F_f_i(p_new,1:N);
        delta = R + gamma * C_p_new - C_p;
        
        w = w + eps * delta * f_i';
        z(action,:) = z(action,:) + eps * delta * f_i';
        
        %prepare next trial
        p = p_new;
        p_logger(tr,t,:) = p;
        h_logger(t,:) = heading/(sqrt(heading(1)^2+heading(2)^2)/dist);
        prev_heading = heading;
        t = t+1;
        
        if t == max_t_steps
            trialend = true; %time-out
        end
    end
    %log the critic after trial for grid x y
    for xi = 1:size(x,2)
        for yi = 1:size(y,2)
            f_i = F_f_i([x(xi),y(yi)],1:N); %firing rates
            C_logger(tr,xi,yi) = sum(w*f_i);            
            
            %preferred action
            a_j = z(:,:)*f_i; %activity of action cell
            P_j = exp(2*a_j)/sum(exp(2*a_j)); %probs of directions
            [pr,i] = max(P_j); %max prob and index of the action
            %logarithm of unit vector times the max prob
            pa_logger(tr,xi,yi,:) = directions(i,:)*pr; 
        end
    end
end
    
%figure 3 of article
figure()
subplot(2,3,1)
surf(x,y,squeeze(C_logger(2,:,:))')
hold on;
plot3(p_pl(1),p_pl(2),C_logger(2,(int64(p_pl(1)+1)/dx),int64((p_pl(2)+1)/dx)),'marker','x','MarkerSize',20)
subplot(2,3,2)
surf(x,y,squeeze(C_logger(7,:,:))')
subplot(2,3,3)
surf(x,y,squeeze(C_logger(22,:,:))')
subplot(2,3,4)
plot(p_logger(2,:,1),p_logger(2,:,2))
hold on;
plot(p_pl(1),p_pl(2),'r.','MarkerSize',25)
axis([-1 1 -1 1])
subplot(2,3,5)
plot(p_logger(7,:,1),p_logger(7,:,2))
hold on;
plot(p_pl(1),p_pl(2),'r.','MarkerSize',25)
axis([-1 1 -1 1])
subplot(2,3,6)
plot(p_logger(22,:,1),p_logger(22,:,2))
hold on;
plot(p_pl(1),p_pl(2),'r.','MarkerSize',25)
axis([-1 1 -1 1])
hold off

%plot some sample paths
figure()
tr = [1 3 5 8 11 14 17 20 21];
for i = 1:9
    subplot(3,3,i)
    plot(p_logger(i,:,1),p_logger(i,:,2))
    axis([-1.1 1.1 -1.1 1.1])
    hold on;
    plot(p_pl(1),p_pl(2),'r.','MarkerSize',20) 
end
hold off

%plot preferred directions
%direction with max probability, vector length the log of that prob
figure()
tr = [2 7 22];
for itr = 1:size(tr,2)
    subplot(1,size(tr,2),itr)
    ind = 1:5:size(x,2);
    quiver(x(ind),y(ind),squeeze(pa_logger(tr(itr),ind,ind,1)),squeeze(pa_logger(tr(itr),ind,ind,2)))
end

