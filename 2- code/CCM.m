function CCM()
% Manopt example on how to use counters during optimization. Typical uses,
% as demonstrated here, include counting calls to cost, gradient and
% Hessian functions. The example also demonstrates how to record total time
% spent in cost/grad/hess calls iteration by iteration.
%
% See also: statscounters incrementcounter statsfunhelper

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 27, 2018.
% Contributors: 
% Change log: 

    addpath("CCM_tool");
    rng(0);
    mode.N = 4;
    mode.Nr = 10;
    mode.Nt = 10;
    % Setup an optimization problem to illustrate the use of counters
    Nt = mode.Nt;
    Nr = mode.Nr;
    N = mode.N;
    K = 3;
    sigma_0 = 1000;
    %sigma_k = [0.5,0.2,0.3];
    sigma_k = [100,100,100];
    sigma_v = 1;
    %sigma_v = 0.8;
    q = sigma_k / sigma_v;
    rk = [0,1,2];
    r0 = 0;
    theta0 = 15;
    theta = [-50,-10,40];
    
    s_init = zeros(Nt,N);
    for k = 1:Nt
        for n = 1:N
            s_init(k,n) = exp(1i * 2 * pi * (n - 1) * (k + n - 1) / N);
        end
    end

% initial f
    s0 = s_init(:);
    clear s_init
    
    A0 = A(theta0,r0,N,Nr,Nt);
    Ak = zeros(N*Nr,N*Nt,K);
    for k = 1:K  
        Ak(:,:,k) = A(theta(k),rk(k),N,Nr,Nt);
    end
    
    manifold = complexcirclefactory(N*Nt);
    problem.M = manifold;
    
    
    % Define the problem cost function and its gradient.
    %
    % Since the most expensive operation in computing the cost and the
    % gradient at x is the product A*x, and since this operation is the
    % same for both the cost and the gradient, we use the caching
    % functionalities of manopt for this product. This function ensures the
    % product A*x is available in the store structure. Remember that a
    % store structure is associated to a particular point x: if cost and
    % egrad are called on the same point x, they will see the same store.
    function store = prepare(x, store)
        store.phi_S = phi(x*x',K,Ak,q,theta,N,Nr);
        store.I = eye(N*Nr);
        store.Ax = A0*x;
        % Increment a counter for the number of matrix-vector products
        % involving A. The names of the counters (here, Aproducts) are
        % for us to choose: they only need to be valid structure field
        % names. They need not have been defined in advance.
        store = incrementcounter(store, 'Aproducts');
    end
    %
    problem.cost = @cost;
    function [f, store] = cost(x, store)
        t = tic();
        store = prepare(x, store);
        f = - store.Ax' / (store.phi_S + store.I) * store.Ax;
        % Increment a counter for the number of calls to the cost function.
        store = incrementcounter(store, 'costcalls');
        % We also increment a counter with the amount of time spent in this
        % function (all counters are stored as doubles; here we exploit
        % this to track a non-integer quantity.)
        store = incrementcounter(store, 'functiontime', toc(t));
    end
    %
    problem.egrad = @egrad;
    function [g, store] = egrad(x, store)
        t = tic();
        store = prepare(x, store);
        g = -fun_grad(x, A0,store.phi_S,K,q,Ak );
        % Count the number of calls to the gradient function.
        store = incrementcounter(store, 'gradcalls');
        % We also record time spent in this call, atop the same counter as
        % for the cost function.
        store = incrementcounter(store, 'functiontime', toc(t));
    end
    %
    problem.sinr = @SINR;
    function [sinr, store] = SINR(x, store)
        t = tic();
        store = prepare(x, store);
        temp = (store.phi_S + store.I);
        filter = (temp \ A0 * x)/(x'*A0'*temp*A0*x);
        numerator = sigma_0 * norm(filter'*A0*x)^2;
        P = phi(x * x',K,Ak,sigma_k,theta,N,Nr);
        denominator = real(filter' * P * filter + sigma_v * (filter' * filter));
        sinr = 10 * log10(numerator / denominator);
        
        % Count the number of calls to the gradient function.
        store = incrementcounter(store, 'sinrcalls');
        % We also record time spent in this call, atop the same counter as
        % for the cost function.
        store = incrementcounter(store, 'functiontime', toc(t));
    end    
%     problem.ehess = @ehess;
%     function [h, store] = ehess(x, xdot, store) %#ok<INUSL>
%         t = tic();
%         h = -A*xdot;
%         % Count the number of calls to the Hessian operator and also count
%         % the matrix-vector product with A.
%         store = incrementcounter(store, 'hesscalls');
%         store = incrementcounter(store, 'Aproducts');
%         % We also record time spent in this call atop the cost and gradient.
%         store = incrementcounter(store, 'functiontime', toc(t));
%     end

    
    % Setup a callback to log statistics. We use a combination of
    % statscounters and of statsfunhelper to indicate which counters we
    % want the optimization algorithm to log. Here, stats is a structure
    % where each field is a function handle corresponding to one of the
    % counters. Before passing stats to statsfunhelper, we could decide to
    % add more fields to stats to log other things as well.
    
    %checkgradient(problem)
    stats = statscounters({'costcalls', 'gradcalls', 'hesscalls', ...
                           'Aproducts', 'functiontime'});
    options.statsfun = statsfunhelper(stats);

    % As an example: we could set up a stopping criterion based on the
    % number of matrix-vector products. A short version:
    % options.stopfun = @(problem, x, info, last) info(last).Aproducts > 250;
    % A longer version that also returns a reason string:
%     options.stopfun = @stopfun;
%     function [stop, reason] = stopfun(problem, x, info, last) %#ok<INUSL>
%         reason = 'Exceeded Aproducts budget.';
%         stop = (info(last).Aproducts > 250);   % true if budget exceeded
%         % Here, info(last) contains the stats of the latest iteration.
%         % That includes all registered counters.
%     end
    options.tolcost = -2.9862089111e+02	;
    options.maxiter = 4;
    
    % Solve with different solvers to compare.
    options.tolgradnorm = 1e-9;
    options.minstepsize = 1e-16;
    options.ls_suff_decr = 1e-6;
    options.ls_max_steps = 25;
    [~, ~, infortr] = steepestdescent(problem, s0, options);
    visualization(infortr)
    temp_t = [infortr.time];
    t_RGD = temp_t(end)
%     [~, ~, infortr] = trustregions(problem, s0, options);
%     visualization(infortr)
%     [~, ~, infortr] = conjugategradient(problem, s0, options);
%     visualization(infortr)
    addpath('D:\SynologyDrive\Research Materials (w Huanyu)\1 - Manifold Opt for Sequence Generation\1 Report\MIMO_Radar_Waveform_Design\CCM');
    
    mode.visualization = true;

    i = 1;
    mode.SimiCon = i == 2;
    mode.PAR = i == 3;
    mode.e = i == 4;
    mode.acceleration = false;
    t_MM=MIA_compare(mode)
    mode.acceleration = true;
    t_MM_SQUARE=MIA_compare(mode)
    
    t_SDR = SDR_compare(mode)
    
    
%    [x, xcost, infortr] = trustregions(problem, s0, options); %#ok<ASGLU>
%     [x, xcost, inforcg] = conjugategradient(problem, s0, options); %#ok<ASGLU>
%     [x, xcost, infobfg] = rlbfgs(problem, s0, options); %#ok<ASGLU>
%     
%     
%     % Display some statistics. The logged data is available in the info
%     % struct-arrays. Notice how the counters are available by their
%     % corresponding field name.
%     figure(1);
%     subplot(3, 3, 1);
%     semilogy([infortr.iter], [infortr.gradnorm], '.-', ...
%              [inforcg.iter], [inforcg.gradnorm], '.-', ...
%              [infobfg.iter], [infobfg.gradnorm], '.-');
%     legend('RTR', 'RCG', 'RLBFGS');
%     xlabel('Iteration #');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     subplot(3, 3, 2);
%     semilogy([infortr.costcalls], [infortr.gradnorm], '.-', ...
%              [inforcg.costcalls], [inforcg.gradnorm], '.-', ...
%              [infobfg.costcalls], [infobfg.gradnorm], '.-');
%     xlabel('# cost calls');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     subplot(3, 3, 3);
%     semilogy([infortr.gradcalls], [infortr.gradnorm], '.-', ...
%              [inforcg.gradcalls], [inforcg.gradnorm], '.-', ...
%              [infobfg.gradcalls], [infobfg.gradnorm], '.-');
%     xlabel('# gradient calls');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     subplot(3, 3, 4);
%     semilogy([infortr.hesscalls], [infortr.gradnorm], '.-', ...
%              [inforcg.hesscalls], [inforcg.gradnorm], '.-', ...
%              [infobfg.hesscalls], [infobfg.gradnorm], '.-');
%     xlabel('# Hessian calls');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     subplot(3, 3, 5);
%     semilogy([infortr.Aproducts], [infortr.gradnorm], '.-', ...
%              [inforcg.Aproducts], [inforcg.gradnorm], '.-', ...
%              [infobfg.Aproducts], [infobfg.gradnorm], '.-');
%     xlabel('# matrix-vector products');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     subplot(3, 3, 6);
%     semilogy([infortr.time], [infortr.gradnorm], '.-', ...
%              [inforcg.time], [inforcg.gradnorm], '.-', ...
%              [infobfg.time], [infobfg.gradnorm], '.-');
%     xlabel('Computation time [s]');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     subplot(3, 3, 7);
%     semilogy([infortr.functiontime], [infortr.gradnorm], '.-', ...
%              [inforcg.functiontime], [inforcg.gradnorm], '.-', ...
%              [infobfg.functiontime], [infobfg.gradnorm], '.-');
%     xlabel('Time spent in cost/grad/hess [s]');
%     ylabel('Gradient norm');
%     ylim([1e-12, 1e2]); set(gca, 'YTick', [1e-12, 1e-6, 1e0]);
%     % The following plot allows to investigate what fraction of the time is
%     % spent inside user-supplied function (cost/grad/hess) versus the total
%     % time spent by the solver. This gives a sense of the relative
%     % importance of cost function-related computational costs vs a solver's
%     % inner workings, retractions, and other solver-specific operations.
%     subplot(3, 3, 8);
%     maxtime = max([[infortr.time], [inforcg.time], [infobfg.time]]);
%     plot([infortr.time], [infortr.functiontime], '.-', ...
%          [inforcg.time], [inforcg.functiontime], '.-', ...
%          [infobfg.time], [infobfg.functiontime], '.-', ...
%          [0, maxtime], [0, maxtime], 'k--');
%     axis tight;
%     xlabel('Total computation time [s]');
%     ylabel(sprintf('Time spent in\ncost/grad/hess [s]'));
    
end
