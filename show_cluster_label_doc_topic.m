
temp_phi = bsxfun(@times, model.beta_dir + model.n_topic_word, 1./(model.beta_dir .* size(model.n_topic_word,2) + sum(model.n_topic_word,2)));
temp_theta = bsxfun(@times, model.alpha + model.n_p_topic, 1./(sum(model.alpha) + sum(model.n_p_topic,2)));
[unique_m, ~, J]=unique(model.pd);
occ = histc(J, 1:numel(unique_m));
occ_idx = find(occ >50);

doc_label_tmp = [doc_label, ones(size(doc_label,1),1)];
pi_dir = exp(doc_label_tmp * log(model.lambda));


for l = 1:size(doc_label,2)-1
    
    [~,ms] = sort(model.lambda(l,:),'descend');
    ms = ms(1:10);
    
    j = 0;
    for m = ms
        j = j + 1;
        
        [~,top_label] = sort(model.lambda(1:end-1,m),'descend');
        fprintf('label: %s, cluster %d\n',strjoin(label_name(top_label(1)),' '), j);
        
        
        m_doc = find(model.pd == m);
        
        [~,top_doc] = sort(pi_dir(m_doc,m),'descend');
        
        tt = min(5,length(top_doc));
        
        top_doc = top_doc(1:tt);
        
        top_doc = m_doc(top_doc);
        
        
        for i = top_doc'
            
            fprintf('doc id: %d, words: %s\n', i, strjoin(voc(find(doc(:,i))),' '));
        end
                
        
        show_top_words_simple(temp_phi,voc, 5, 0, temp_theta(m,:),5);
        fprintf('==============================================================================\n');
    end
    
end
