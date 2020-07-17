
function show_top_words_simple(phi, voc, top_W, out_file, topic_weight, top_K)


if ~exist('out_file','var')
    out_file = false;
end

[~, sorted_tw_idx] = sort(phi,2, 'descend');


K = size(phi,1);

if ~exist('top_K', 'var')
   top_K = K; 
end


if exist('topic_weight','var')
    [~,sorted_topic_idx] = sort(topic_weight, 'descend');
else
    sorted_topic_idx = 1:K;
end


for k = 1:top_K
    
    top_words = [];
    for v = 1:top_W
        top_words = [top_words, ' ', voc{sorted_tw_idx(sorted_topic_idx(k),v)}];
    end

    if ~out_file
        fprintf('topic id: %d, top words: %s\n', sorted_topic_idx(k), top_words);
    else
        fprintf(out_file,'%s\n',top_words);
    end
end


