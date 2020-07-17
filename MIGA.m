function model = MIGA(X_all, doc_label, Para)

    X = X_all;
    [V,D] = size(X);
    [Xtrain, Xtest, WS, DS, WordTrainS, DocTrainS] = PartitionX_v1(X,101);

    WS = WS(WordTrainS); 
    DS = DS(WordTrainS); 
    ZS = DS-DS;

    alpha = ones(1,Para.K) * 0.1;
    sum_alpha = sum(alpha);

    if Para.is_meta_data > 0

        doc_label = [doc_label, ones(size(doc_label,1),1)];
        L = size(doc_label,2);
        mu_d = 1;
        nu_d_0 = 1 ./ mu_d;
        lambda = 1.0 * ones(L,Para.M);
        pi_dir = exp(doc_label * log(lambda));
        sum_pi_dir = sum(pi_dir,2);
        d_f_idx = cell(L);

        for l = 1:L
            d_f_idx{l} = find(doc_label(:,l));
        end
    else
        lambda = 0.1;    
        pi_dir = NaN;
    end

    beta_dir = 0.01;
    sum_beta_dir = beta_dir * V;
    n_doc_topic = zeros(D,Para.K);
    n_doc_dot = zeros(D,1);
    n_topic_word = zeros(Para.K,V);
    n_topic_dot = zeros(Para.K,1);
    n_p_topic = zeros(Para.M,Para.K);
    n_p_dot = zeros(Para.M,1);
    pd = randi(Para.M,D,1); % which pseudo-doc
    [n_p_doc,~]=hist(pd,1:Para.M);
    
    for i = 1:length(DS)
        d = DS(i);
        v = WS(i);
        m_d = pd(d);
        k = randi(Para.K, 1);
        n_topic_word(k,v) = n_topic_word(k,v) + 1;
        n_topic_dot(k) = n_topic_dot(k) + 1;
        n_p_topic(m_d,k) = n_p_topic(m_d,k) + 1;
        n_p_dot(m_d) = n_p_dot(m_d) + 1;
        n_doc_topic(d,k) = n_doc_topic(d,k) + 1;
        n_doc_dot(d) = n_doc_dot(d) + 1;
        ZS(i) = k;
    end

    for r = 1:Para.max_iter

        tic;    
        for d = 1:D

            m_d = pd(d);
            n_p_topic(m_d,:) = n_p_topic(m_d,:) - n_doc_topic(d,:);

            n_p_dot(m_d) = n_p_dot(m_d) - n_doc_dot(d);
            n_p_doc(m_d) = n_p_doc(m_d) - 1;

            if Para.is_meta_data > 0
                p1 = pi_dir(d,:) ./ sum_pi_dir(d);
            else
                p1 = n_p_doc + lambda;
            end

            p2 = ones(1,Para.M);
            p3 = ones(1,Para.M);

            for k = find(n_doc_topic(d,:))
                for j = 1: n_doc_topic(d,k)
                    p2 = p2 .* (alpha(k) + n_p_topic(:,k)' + j - 1);
                end
            end

            for i = 1:n_doc_dot(d)
                p3 = p3 .* (sum_alpha + n_p_dot' + i - 1);
            end
            p = p1 .* p2 ./ p3;
            sum_cum = cumsum(p(:));
            new_m_d = find(sum_cum > rand() * sum_cum(end),1);
            pd(d) = new_m_d;
            n_p_topic(new_m_d,:) = n_p_topic(new_m_d,:) + n_doc_topic(d,:);
            n_p_dot(new_m_d) = n_p_dot(new_m_d) + n_doc_dot(d);
            n_p_doc(new_m_d) = n_p_doc(new_m_d) + 1;

        end
        
        for i = 1:length(DS)
            d = DS(i);
            v = WS(i);
            k = ZS(i);
            m_d = pd(d);
            n_p_topic(m_d,k) = n_p_topic(m_d,k) - 1;
            n_topic_word(k,v) = n_topic_word(k,v) - 1;
            n_topic_dot(k) = n_topic_dot(k) - 1;
            n_doc_topic(d,k) = n_doc_topic(d,k) - 1;
            p_left = (alpha + n_p_topic(m_d,:));
            p_right = (beta_dir + n_topic_word(:,v)) ./ (sum_beta_dir + n_topic_dot);
            p = (p_left .* p_right');
            sum_cum = cumsum(p(:));
            new_k = find(sum_cum > rand() * sum_cum(end),1);
            n_p_topic(m_d,new_k) = n_p_topic(m_d,new_k) + 1;
            n_topic_word(new_k,v) = n_topic_word(new_k,v) + 1;
            n_topic_dot(new_k) = n_topic_dot(new_k) + 1;
            n_doc_topic(d,new_k) = n_doc_topic(d,new_k) + 1;
            ZS(i) = new_k;

        end

        if Para.is_meta_data > 0
            n_doc_pdoc = zeros(D, Para.M);
            n_doc_pdoc(sub2ind([D,Para.M],1:D, pd')) = 1;
            d_log_q = -log(betarnd(sum_pi_dir,sum(n_doc_pdoc,2)));
            sum_label_p = doc_label' * n_doc_pdoc;
            new_lambda = randg(mu_d + sum_label_p);

            for l = 1:L

                p_h = d_f_idx{l};
                p_pi = sum(bsxfun(@times,pi_dir(p_h,:),d_log_q(p_h)));
                new_lambda_l = new_lambda(l,:) ./ (p_pi + nu_d_0 * lambda(l,:));

                pi_dir(p_h,:) = bsxfun(@times,pi_dir(p_h,:),new_lambda_l);
                lambda(l,:) = new_lambda_l .* lambda(l,:);
            end
            sum_pi_dir = sum(pi_dir,2);
        end

        if Para.is_sample_alpha
            alpha = sample_asym_dir_hyper_row(alpha, n_p_topic);
            sum_alpha = sum(alpha);
        end

        Timetmp = toc;

        if mod(r,10) == 0
            if Para.is_meta_data > 0
                log_ll = compute_log_likelihood(D, Para.M, Para.K, V, pd, pi_dir, alpha, n_p_topic, n_p_dot, beta_dir, n_topic_word, n_topic_dot);
            else
                log_ll = compute_log_likelihood(D, Para.M, Para.K, V, pd, [], alpha, n_p_topic, n_p_dot, beta_dir, n_topic_word, n_topic_dot);
            end
            fprintf('iter: %d, time: %0.1f, log likelihood: %f\n', r, Timetmp, log_ll);
        end
        
    end

    model.lambda = lambda;
    model.pd = pd;
    model.alpha = alpha;
    model.beta_dir = beta_dir;
    model.n_doc_topic = sparse(n_doc_topic);
    model.n_topic_word = sparse(n_topic_word);
    model.n_p_topic = sparse(n_p_topic);
    model.Para = Para;
    model.ZS = ZS;

    if exist('mu_d','var')
        model.mu_d = mu_d;
    end

end



function alpha = sample_asym_dir_hyper_row(alpha, n) % all rows are same

    mu_0 = 1.0;
    nu_0 = 1.0;
    [N,K] = size(n);
    log_q = -log(betarnd(sum(alpha), sum(n,2)));
    t = zeros(N,K);
    t(n>0) = 1;
    for i = 1:N
        for k = 1:K
            for j=1:n(i,k)-1
                t(i,k) = t(i,k) + double(rand() < alpha(k) ./ (alpha(k) + j));
            end
        end
    end

    alpha = randg(mu_0 + sum(t,1)) ./ (nu_0 + sum(log_q));

end

function log_ll = compute_log_likelihood(D, M, K, V, pd, pi, alpha, n_p_topic, n_p_dot, beta_dir, n_topic_word, n_topic_dot)

    log_ll = 0;

    if ~isempty(pi)
        for d = 1:D
            m_d = pd(d);
            log_ll = log_ll + log(pi(d, m_d)) - log(sum(pi(d,:)));

        end
    end

    sum_alpha = sum(alpha);
    log_ll = log_ll + M * gammaln(sum_alpha) - sum( gammaln(sum_alpha + n_p_dot));


    log_ll = log_ll + sum(gammaln(alpha + n_p_topic),'all') - M *  sum(gammaln(alpha));

    log_ll = log_ll + K * gammaln(beta_dir * V) - sum(gammaln(beta_dir * V + n_topic_dot));

    log_ll = log_ll + sum(gammaln(beta_dir + n_topic_word), 'all') - K * V * gammaln(beta_dir);



end




